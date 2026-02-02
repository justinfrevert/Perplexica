import { ActionOutput, ResearcherInput, ResearcherOutput } from '../types';
import { ActionRegistry } from './actions';
import { getResearcherPrompt } from '@/lib/prompts/search/researcher';
import SessionManager from '@/lib/session';
import { Chunk, Message, ReasoningResearchBlock } from '@/lib/types';
import formatChatHistoryAsString from '@/lib/utils/formatHistory';
import { ToolCall } from '@/lib/models/types';
import { createSearchDebugLogger } from '../debug';
import { getCrawl4aiURLs } from '@/lib/config/serverRegistry';

const runWithConcurrency = async <T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
): Promise<T[]> => {
  if (tasks.length === 0) {
    return [];
  }

  const results: T[] = new Array(tasks.length);
  let nextIndex = 0;

  const worker = async () => {
    while (nextIndex < tasks.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await tasks[currentIndex]();
    }
  };

  const workerCount = Math.min(Math.max(concurrency, 1), tasks.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  return results;
};

const hasCrawl4aiToolOutput = (messages: Message[]) =>
  messages.some(
    (message) => message.role === 'tool' && message.name === 'scrape_url',
  );

const logCrawl4aiPrompt = (
  log: (...args: unknown[]) => void,
  stage: string,
  messages: Message[],
  meta: Record<string, unknown>,
) => {
  const rawPrompt = JSON.stringify(messages);
  log('llm:prompt:crawl4ai:raw', { stage, ...meta }, rawPrompt);
  log('llm:prompt:crawl4ai:length', {
    stage,
    ...meta,
    length: rawPrompt.length,
  });
};

class Researcher {
  async research(
    session: SessionManager,
    input: ResearcherInput,
  ): Promise<ResearcherOutput> {
    const log = createSearchDebugLogger(input.debugSessionId ?? session.id);
    let actionOutput: ActionOutput[] = [];
    let maxIteration =
      input.config.mode === 'speed'
        ? 2
        : input.config.mode === 'balanced'
          ? 6
          : 25;

    const availableTools = ActionRegistry.getAvailableActionTools({
      classification: input.classification,
      fileIds: input.config.fileIds,
      mode: input.config.mode,
      sources: input.config.sources,
    });

    const availableActionsDescription =
      ActionRegistry.getAvailableActionsDescriptions({
        classification: input.classification,
        fileIds: input.config.fileIds,
        mode: input.config.mode,
        sources: input.config.sources,
      });

    const researchBlockId = crypto.randomUUID();

    log('research:start', {
      mode: input.config.mode,
      maxIteration,
      sources: input.config.sources,
      fileIds: input.config.fileIds.length,
      tools: availableTools.map((tool) => tool.name),
    });

    session.emitBlock({
      id: researchBlockId,
      type: 'research',
      data: {
        subSteps: [],
      },
    });

    const agentMessageHistory: Message[] = [
      {
        role: 'user',
        content: `
          <conversation>
          ${formatChatHistoryAsString(input.chatHistory.slice(-10))}
           User: ${input.followUp} (Standalone question: ${input.classification.standaloneFollowUp})
           </conversation>
        `,
      },
    ];

    for (let i = 0; i < maxIteration; i++) {
      const iteration = i + 1;
      log('research:iteration:start', { iteration, maxIteration });
      const researcherPrompt = getResearcherPrompt(
        availableActionsDescription,
        input.config.mode,
        i,
        maxIteration,
        input.config.fileIds,
      );

      const promptMessages: Message[] = [
        {
          role: 'system',
          content: researcherPrompt,
        },
        ...agentMessageHistory,
      ];

      if (hasCrawl4aiToolOutput(promptMessages)) {
        logCrawl4aiPrompt(log, 'researcher', promptMessages, {
          iteration,
          maxIteration,
        });
      }

      const actionStream = input.config.llm.streamText({
        messages: promptMessages,
        tools: availableTools,
      });

      const block = session.getBlock(researchBlockId);

      let reasoningEmitted = false;
      let reasoningId = crypto.randomUUID();

      let finalToolCalls: ToolCall[] = [];

      for await (const partialRes of actionStream) {
        if (partialRes.toolCallChunk.length > 0) {
          partialRes.toolCallChunk.forEach((tc) => {
            if (
              tc.name === '__reasoning_preamble' &&
              tc.arguments['plan'] &&
              !reasoningEmitted &&
              block &&
              block.type === 'research'
            ) {
              reasoningEmitted = true;

              block.data.subSteps.push({
                id: reasoningId,
                type: 'reasoning',
                reasoning: tc.arguments['plan'],
              });

              session.updateBlock(researchBlockId, [
                {
                  op: 'replace',
                  path: '/data/subSteps',
                  value: block.data.subSteps,
                },
              ]);
            } else if (
              tc.name === '__reasoning_preamble' &&
              tc.arguments['plan'] &&
              reasoningEmitted &&
              block &&
              block.type === 'research'
            ) {
              const subStepIndex = block.data.subSteps.findIndex(
                (step: any) => step.id === reasoningId,
              );

              if (subStepIndex !== -1) {
                const subStep = block.data.subSteps[
                  subStepIndex
                ] as ReasoningResearchBlock;
                subStep.reasoning = tc.arguments['plan'];
                session.updateBlock(researchBlockId, [
                  {
                    op: 'replace',
                    path: '/data/subSteps',
                    value: block.data.subSteps,
                  },
                ]);
              }
            }

            const existingIndex = finalToolCalls.findIndex(
              (ftc) => ftc.id === tc.id,
            );

            if (existingIndex !== -1) {
              finalToolCalls[existingIndex].arguments = tc.arguments;
            } else {
              finalToolCalls.push(tc);
            }
          });
        }
      }

      log('research:iteration:tool_calls', {
        iteration,
        count: finalToolCalls.length,
        tools: Array.from(new Set(finalToolCalls.map((tc) => tc.name))),
      });

      if (finalToolCalls.length === 0) {
        log('research:iteration:end', { iteration, reason: 'no_tool_calls' });
        break;
      }

      if (finalToolCalls[finalToolCalls.length - 1].name === 'done') {
        log('research:iteration:end', { iteration, reason: 'done' });
        break;
      }

      agentMessageHistory.push({
        role: 'assistant',
        content: '',
        tool_calls: finalToolCalls,
      });

      log('research:actions:start', {
        iteration,
        actions: finalToolCalls.map((tc) => tc.name),
      });

      const actionResults = await ActionRegistry.executeAll(finalToolCalls, {
        llm: input.config.llm,
        embedding: input.config.embedding,
        session: session,
        researchBlockId: researchBlockId,
        fileIds: input.config.fileIds,
      });

      actionOutput.push(...actionResults);

      log('research:actions:done', {
        iteration,
        actionResults: actionResults.length,
        totalOutputs: actionOutput.length,
      });

      actionResults.forEach((action, i) => {
        agentMessageHistory.push({
          role: 'tool',
          id: finalToolCalls[i].id,
          name: finalToolCalls[i].name,
          content: JSON.stringify(action),
        });
      });
    }

    const collectSearchResults = (outputs: ActionOutput[]) =>
      outputs
        .filter((a) => a.type === 'search_results')
        .flatMap((a) => a.results);

    const isHttpUrl = (url: string) =>
      url.startsWith('http://') || url.startsWith('https://');

    const isScrapeResult = (result: Chunk) =>
      result.metadata?.source === 'scrape_url' || result.metadata?.scraped === true;

    const hasScrapeError = (result: Chunk) =>
      Boolean(result.metadata?.scrapeError);

    const getResultRank = (result: Chunk) => {
      if (isScrapeResult(result)) {
        return hasScrapeError(result) ? 0 : 2;
      }

      return 1;
    };

    const isFileResult = (result: Chunk) => {
      const url = result.metadata?.url;

      if (typeof url === 'string' && url.startsWith('file_id://')) {
        return true;
      }

      return Boolean(result.metadata?.fileId);
    };

    const isCitableResult = (result: Chunk) => {
      if (isFileResult(result)) {
        return true;
      }

      return isScrapeResult(result) && !hasScrapeError(result);
    };

    const initialSearchResults = collectSearchResults(actionOutput);
    const urlsToScrape: string[] = [];
    const seenScrapeUrls = new Set<string>();

    initialSearchResults.forEach((result) => {
      const url = result.metadata?.url;

      if (!url || !isHttpUrl(url) || isScrapeResult(result)) {
        return;
      }

      if (!seenScrapeUrls.has(url)) {
        seenScrapeUrls.add(url);
        urlsToScrape.push(url);
      }
    });

    if (urlsToScrape.length > 0 && ActionRegistry.get('scrape_url')) {
      log('research:scrape:queue', { urls: urlsToScrape.length });
      const totalBatches = Math.ceil(urlsToScrape.length / 3);
      const batches = Array.from({ length: totalBatches }, (_, batchOffset) => {
        const start = batchOffset * 3;
        const batch = urlsToScrape.slice(start, start + 3);

        return {
          batch,
          batchIndex: batchOffset + 1,
          totalBatches,
        };
      });

      const crawl4aiEndpoints = getCrawl4aiURLs();
      const scrapeConcurrency = Math.min(
        Math.max(crawl4aiEndpoints.length, 1),
        batches.length,
      );

      const outputs = await runWithConcurrency(
        batches.map(({ batch, batchIndex, totalBatches }) => async () => {
          const startTime = Date.now();

          try {
            log('research:scrape:batch:start', {
              batchIndex,
              totalBatches,
              batchSize: batch.length,
            });
            const output = await ActionRegistry.execute(
              'scrape_url',
              { urls: batch },
              {
                llm: input.config.llm,
                embedding: input.config.embedding,
                session,
                researchBlockId,
                fileIds: input.config.fileIds,
              },
            );
            log('research:scrape:batch:done', {
              batchIndex,
              durationMs: Date.now() - startTime,
            });
            return output;
          } catch (error) {
            console.log('[researcher] scrape_url failed', { batch, error });
            log('research:scrape:batch:error', {
              batchIndex,
              error: error instanceof Error ? error.message : String(error),
            });
            return null;
          }
        }),
        scrapeConcurrency,
      );

      outputs.forEach((output) => {
        if (output) {
          actionOutput.push(output);
        }
      });
    }

    const searchResults = collectSearchResults(actionOutput);
    const filteredSearchResults: Chunk[] = [];
    const seenUrls = new Map<string, number>();

    searchResults.forEach((result) => {
      const url = result.metadata?.url;

      if (!url) {
        filteredSearchResults.push(result);
        return;
      }

      const existingIndex = seenUrls.get(url);
      if (existingIndex === undefined) {
        seenUrls.set(url, filteredSearchResults.length);
        filteredSearchResults.push(result);
        return;
      }

      const existingResult = filteredSearchResults[existingIndex];
      const existingRank = getResultRank(existingResult);
      const incomingRank = getResultRank(result);

      if (incomingRank > existingRank) {
        filteredSearchResults[existingIndex] = result;
        return;
      }

      if (incomingRank === existingRank) {
        if (incomingRank === 2) {
          return;
        }

        existingResult.content += `\n\n${result.content}`;
      }
    });

    const citableResults = filteredSearchResults.filter(isCitableResult);
    const lightResults = filteredSearchResults.filter(
      (result) => !isCitableResult(result),
    );

    log('research:results:final', {
      initialResults: initialSearchResults.length,
      filteredResults: filteredSearchResults.length,
      citableResults: citableResults.length,
      lightResults: lightResults.length,
    });

    if (citableResults.length > 0) {
      session.emitBlock({
        id: crypto.randomUUID(),
        type: 'source',
        data: citableResults,
      });
    }

    if (lightResults.length > 0) {
      session.emitBlock({
        id: crypto.randomUUID(),
        type: 'source_light',
        data: lightResults,
      });
    }

    log('research:done');

    return {
      findings: actionOutput,
      searchFindings: citableResults,
      lightSearchFindings: lightResults,
    };
  }
}

export default Researcher;

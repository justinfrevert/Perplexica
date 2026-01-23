import { ActionOutput, ResearcherInput, ResearcherOutput } from '../types';
import { ActionRegistry } from './actions';
import { getResearcherPrompt } from '@/lib/prompts/search/researcher';
import SessionManager from '@/lib/session';
import { Chunk, Message, ReasoningResearchBlock } from '@/lib/types';
import formatChatHistoryAsString from '@/lib/utils/formatHistory';
import { ToolCall } from '@/lib/models/types';

class Researcher {
  async research(
    session: SessionManager,
    input: ResearcherInput,
  ): Promise<ResearcherOutput> {
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
      const researcherPrompt = getResearcherPrompt(
        availableActionsDescription,
        input.config.mode,
        i,
        maxIteration,
        input.config.fileIds,
      );

      const actionStream = input.config.llm.streamText({
        messages: [
          {
            role: 'system',
            content: researcherPrompt,
          },
          ...agentMessageHistory,
        ],
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

      if (finalToolCalls.length === 0) {
        break;
      }

      if (finalToolCalls[finalToolCalls.length - 1].name === 'done') {
        break;
      }

      agentMessageHistory.push({
        role: 'assistant',
        content: '',
        tool_calls: finalToolCalls,
      });

      const actionResults = await ActionRegistry.executeAll(finalToolCalls, {
        llm: input.config.llm,
        embedding: input.config.embedding,
        session: session,
        researchBlockId: researchBlockId,
        fileIds: input.config.fileIds,
      });

      actionOutput.push(...actionResults);

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
      for (let i = 0; i < urlsToScrape.length; i += 3) {
        const batch = urlsToScrape.slice(i, i + 3);

        try {
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
          actionOutput.push(output);
        } catch (error) {
          console.log('[researcher] scrape_url failed', { batch, error });
        }
      }
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

    return {
      findings: actionOutput,
      searchFindings: citableResults,
      lightSearchFindings: lightResults,
    };
  }
}

export default Researcher;

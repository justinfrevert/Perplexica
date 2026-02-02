import { ResearcherOutput, SearchAgentInput } from './types';
import SessionManager from '@/lib/session';
import { classify } from './classifier';
import Researcher from './researcher';
import { getWriterPrompt } from '@/lib/prompts/search/writer';
import { WidgetExecutor } from './widgets';
import { Chunk, Message } from '@/lib/types';
import { createSearchDebugLogger } from './debug';

const hasCrawl4aiResults = (results: Chunk[] | undefined) =>
  Boolean(
    results?.some(
      (result) =>
        result.metadata?.source === 'scrape_url' ||
        result.metadata?.scraped === true,
    ),
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

class APISearchAgent {
  async searchAsync(session: SessionManager, input: SearchAgentInput) {
    const log = createSearchDebugLogger(session.id);

    log('agent:start', {
      sources: input.config.sources,
      optimizationMode: input.config.mode,
      fileIds: input.config.fileIds.length,
      queryLength: input.followUp.length,
      historyLength: input.chatHistory.length,
    });

    const classification = await classify({
      chatHistory: input.chatHistory,
      enabledSources: input.config.sources,
      query: input.followUp,
      llm: input.config.llm,
    });

    log('classification:done', {
      skipSearch: classification.classification.skipSearch,
      personalSearch: classification.classification.personalSearch,
      academicSearch: classification.classification.academicSearch,
      discussionSearch: classification.classification.discussionSearch,
      showWeatherWidget: classification.classification.showWeatherWidget,
      showStockWidget: classification.classification.showStockWidget,
      showCalculationWidget: classification.classification.showCalculationWidget,
      standaloneLength: classification.standaloneFollowUp.length,
    });

    log('widgets:start');
    const widgetPromise = WidgetExecutor.executeAll({
      classification,
      chatHistory: input.chatHistory,
      followUp: input.followUp,
      llm: input.config.llm,
      sessionId: session.id,
    }).then((widgetOutputs) => {
      log('widgets:done', { widgets: widgetOutputs.length });
      return widgetOutputs;
    });

    let searchPromise: Promise<ResearcherOutput> | null = null;

    if (!classification.classification.skipSearch) {
      log('research:start');
      const researcher = new Researcher();
      searchPromise = researcher.research(SessionManager.createSession(), {
        chatHistory: input.chatHistory,
        followUp: input.followUp,
        classification: classification,
        config: input.config,
        debugSessionId: session.id,
      });
    } else {
      log('research:skip');
    }

    const [widgetOutputs, searchResults] = await Promise.all([
      widgetPromise,
      searchPromise,
    ]);

    log('research:done', {
      searchFindings: searchResults?.searchFindings?.length ?? 0,
      lightSearchFindings: searchResults?.lightSearchFindings?.length ?? 0,
    });

    if (searchResults) {
      session.emit('data', {
        type: 'searchResults',
        data: {
          sources: searchResults.searchFindings,
          lightSources: searchResults.lightSearchFindings,
        },
      });
    }

    session.emit('data', {
      type: 'researchComplete',
    });

    const finalContext =
      searchResults?.searchFindings
        .map(
          (f, index) =>
            `<result index=${index + 1} title=${f.metadata.title}>${f.content}</result>`,
        )
        .join('\n') || '';

    const widgetContext = widgetOutputs
      .map((o) => {
        return `<result>${o.llmContext}</result>`;
      })
      .join('\n-------------\n');

    log('writer:start', {
      searchContextLength: finalContext.length,
      widgetContextLength: widgetContext.length,
    });

    const finalContextWithWidgets = `<search_results note="These are the search results and assistant can cite these">\n${finalContext}\n</search_results>\n<widgets_result noteForAssistant="Its output is already showed to the user, assistant can use this information to answer the query but do not CITE this as a souce">\n${widgetContext}\n</widgets_result>`;

    const writerPrompt = getWriterPrompt(
      finalContextWithWidgets,
      input.config.systemInstructions,
      input.config.mode,
    );

    const writerMessages: Message[] = [
      {
        role: 'system',
        content: writerPrompt,
      },
      ...input.chatHistory,
      {
        role: 'user',
        content: input.followUp,
      },
    ];

    if (hasCrawl4aiResults(searchResults?.searchFindings)) {
      logCrawl4aiPrompt(log, 'writer_api', writerMessages, {
        searchFindings: searchResults?.searchFindings?.length ?? 0,
      });
    }

    const answerStream = input.config.llm.streamText({
      messages: writerMessages,
    });

    let responseChunks = 0;
    let responseChars = 0;

    for await (const chunk of answerStream) {
      responseChunks += 1;
      if (typeof chunk.contentChunk === 'string') {
        responseChars += chunk.contentChunk.length;
      }
      if (responseChunks === 1 || responseChunks % 50 === 0) {
        log('response:progress', {
          chunks: responseChunks,
          chars: responseChars,
        });
      }

      session.emit('data', {
        type: 'response',
        data: chunk.contentChunk,
      });
    }

    log('writer:done', {
      responseChunks,
      responseChars,
    });

    session.emit('end', {});
  }
}

export default APISearchAgent;

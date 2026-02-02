import ModelRegistry from '@/lib/models/registry';
import { ModelWithProvider } from '@/lib/models/types';
import SessionManager from '@/lib/session';
import { ChatTurnMessage } from '@/lib/types';
import { SearchSources } from '@/lib/agents/search/types';
import APISearchAgent from '@/lib/agents/search/api';
import { createSearchDebugLogger, isSearchDebugEnabled } from '@/lib/agents/search/debug';

interface ChatRequestBody {
  optimizationMode: 'speed' | 'balanced' | 'quality';
  sources: SearchSources[];
  chatModel: ModelWithProvider;
  embeddingModel: ModelWithProvider;
  query: string;
  history: Array<[string, string]>;
  stream?: boolean;
  systemInstructions?: string;
}

export const POST = async (req: Request) => {
  try {
    const body: ChatRequestBody = await req.json();
    const debugSearch = isSearchDebugEnabled();

    if (!body.sources || !body.query) {
      if (debugSearch) {
        console.log('[search] request:invalid', {
          hasSources: !!body.sources,
          hasQuery: !!body.query,
        });
      }
      return Response.json(
        { message: 'Missing sources or query' },
        { status: 400 },
      );
    }

    body.history = body.history || [];
    body.optimizationMode = body.optimizationMode || 'speed';
    body.stream = body.stream || false;

    const registry = new ModelRegistry();

    const [llm, embeddings] = await Promise.all([
      registry.loadChatModel(body.chatModel.providerId, body.chatModel.key),
      registry.loadEmbeddingModel(
        body.embeddingModel.providerId,
        body.embeddingModel.key,
      ),
    ]);

    const history: ChatTurnMessage[] = body.history.map((msg) => {
      return msg[0] === 'human'
        ? { role: 'user', content: msg[1] }
        : { role: 'assistant', content: msg[1] };
    });

    const session = SessionManager.createSession();
    const log = createSearchDebugLogger(session.id, debugSearch);

    log('request:start', {
      stream: body.stream,
      sources: body.sources,
      optimizationMode: body.optimizationMode,
      queryLength: body.query.length,
      historyLength: body.history.length,
    });

    const agent = new APISearchAgent();

    agent.searchAsync(session, {
      chatHistory: history,
      config: {
        embedding: embeddings,
        llm: llm,
        sources: body.sources,
        mode: body.optimizationMode,
        fileIds: [],
        systemInstructions: body.systemInstructions || '',
      },
      followUp: body.query,
      chatId: crypto.randomUUID(),
      messageId: crypto.randomUUID(),
    });

    if (!body.stream) {
      return new Promise(
        (
          resolve: (value: Response) => void,
          reject: (value: Response) => void,
        ) => {
          let message = '';
          let sources: any[] = [];
          let lightSources: any[] = [];
          let responseChunks = 0;
          let responseChars = 0;

          session.subscribe((event: string, data: Record<string, any>) => {
            if (event === 'data') {
              try {
                if (data.type === 'response') {
                  message += data.data;
                  responseChunks += 1;
                  if (typeof data.data === 'string') {
                    responseChars += data.data.length;
                  }
                  if (responseChunks === 1 || responseChunks % 50 === 0) {
                    log('response:progress', {
                      chunks: responseChunks,
                      chars: responseChars,
                    });
                  }
                } else if (data.type === 'searchResults') {
                  if (Array.isArray(data.data)) {
                    sources = data.data;
                    lightSources = [];
                  } else {
                    sources = data.data?.sources || [];
                    lightSources = data.data?.lightSources || [];
                  }
                  log('search:results', {
                    sources: sources.length,
                    lightSources: lightSources.length,
                  });
                } else if (data.type === 'researchComplete') {
                  log('research:complete');
                }
              } catch (error) {
                reject(
                  Response.json(
                    { message: 'Error parsing data' },
                    { status: 500 },
                  ),
                );
              }
            }

            if (event === 'end') {
              log('request:done', {
                responseChunks,
                responseChars,
              });
              resolve(
                Response.json(
                  { message, sources, lightSources },
                  { status: 200 },
                ),
              );
            }

            if (event === 'error') {
              log('request:error', { error: data });
              reject(
                Response.json(
                  { message: 'Search error', error: data },
                  { status: 500 },
                ),
              );
            }
          });
        },
      );
    }

    const encoder = new TextEncoder();

    const abortController = new AbortController();
    const { signal } = abortController;

    const stream = new ReadableStream({
      start(controller) {
        let sources: any[] = [];
        let lightSources: any[] = [];
        let responseChunks = 0;
        let responseChars = 0;

        controller.enqueue(
          encoder.encode(
            JSON.stringify({
              type: 'init',
              data: 'Stream connected',
            }) + '\n',
          ),
        );

        signal.addEventListener('abort', () => {
          log('stream:abort');
          session.removeAllListeners();

          try {
            controller.close();
          } catch (error) {}
        });

        session.subscribe((event: string, data: Record<string, any>) => {
          if (event === 'data') {
            if (signal.aborted) return;

            try {
              if (data.type === 'response') {
                responseChunks += 1;
                if (typeof data.data === 'string') {
                  responseChars += data.data.length;
                }
                if (responseChunks === 1 || responseChunks % 50 === 0) {
                  log('response:progress', {
                    chunks: responseChunks,
                    chars: responseChars,
                  });
                }
                controller.enqueue(
                  encoder.encode(
                    JSON.stringify({
                      type: 'response',
                      data: data.data,
                    }) + '\n',
                  ),
                );
              } else if (data.type === 'searchResults') {
                if (Array.isArray(data.data)) {
                  sources = data.data;
                  lightSources = [];
                } else {
                  sources = data.data?.sources || [];
                  lightSources = data.data?.lightSources || [];
                }
                log('search:results', {
                  sources: sources.length,
                  lightSources: lightSources.length,
                });
                controller.enqueue(
                  encoder.encode(
                    JSON.stringify({
                      type: 'sources',
                      data: sources,
                    }) + '\n',
                  ),
                );
                if (lightSources.length > 0) {
                  controller.enqueue(
                    encoder.encode(
                      JSON.stringify({
                        type: 'light_sources',
                        data: lightSources,
                      }) + '\n',
                    ),
                  );
                }
              } else if (data.type === 'researchComplete') {
                log('research:complete');
              }
            } catch (error) {
              controller.error(error);
            }
          }

          if (event === 'end') {
            if (signal.aborted) return;

            log('request:done', {
              responseChunks,
              responseChars,
            });
            controller.enqueue(
              encoder.encode(
                JSON.stringify({
                  type: 'done',
                }) + '\n',
              ),
            );
            controller.close();
          }

          if (event === 'error') {
            if (signal.aborted) return;

            log('request:error', { error: data });
            controller.error(data);
          }
        });
      },
      cancel() {
        abortController.abort();
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
      },
    });
  } catch (err: any) {
    console.error(`Error in getting search results: ${err.message}`);
    return Response.json(
      { message: 'An error has occurred.' },
      { status: 500 },
    );
  }
};

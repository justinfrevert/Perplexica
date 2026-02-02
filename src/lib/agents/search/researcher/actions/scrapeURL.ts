import z from 'zod';
import { ResearchAction } from '../../types';
import { Chunk } from '@/lib/types';
import TurnDown from 'turndown';
import { getCrawl4aiURLs } from '@/lib/config/serverRegistry';

const turndownService = new TurnDown();
const RETRY_BASE_DELAY_MS = 1500;
const MAX_RETRIES_PER_URL = 4;
let crawl4aiEndpointIndex = 0;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const isRetryableStatus = (status: number) => status >= 500 && status <= 599;

const pickCrawl4aiEndpoint = (endpoints: string[]) => {
  if (endpoints.length === 0) {
    return 'http://localhost:11235/crawl-lite';
  }

  if (endpoints.length === 1) {
    return endpoints[0];
  }

  const index = crawl4aiEndpointIndex % endpoints.length;
  crawl4aiEndpointIndex += 1;

  return endpoints[index];
};

const fetchCrawl4AiWithRetry = async (
  endpoint: string,
  payload: Record<string, unknown>,
) => {
  for (let attempt = 0; attempt <= MAX_RETRIES_PER_URL; attempt += 1) {
    let response: Response;

    try {
      response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
    } catch (error) {
      if (attempt < MAX_RETRIES_PER_URL) {
        const delayMs = RETRY_BASE_DELAY_MS * 2 ** attempt;
        await sleep(delayMs);
        continue;
      }

      throw error;
    }

    if (response.ok) {
      return response;
    }

    if (isRetryableStatus(response.status) && attempt < MAX_RETRIES_PER_URL) {
      const delayMs = RETRY_BASE_DELAY_MS * 2 ** attempt;
      await sleep(delayMs);
      continue;
    }

    throw new Error(
      `crawl4ai request failed: ${endpoint} ${response.status} ${response.statusText}`,
    );
  }

  throw new Error(`crawl4ai request failed after retries: ${endpoint}`);
};

type Crawl4AIResult = {
  url?: string;
  source_url?: string;
  original_url?: string;
  final_url?: string;
  title?: string;
  page_title?: string;
  metadata?: { title?: string };
  meta?: { title?: string };
  markdown?: string;
  content?: string;
  text?: string;
  extracted_text?: string;
  cleaned_html?: string;
  html?: string;
};

type Crawl4AIResponse = {
  results?: Crawl4AIResult[] | Record<string, Crawl4AIResult>;
  data?: Crawl4AIResult[] | Record<string, Crawl4AIResult>;
  items?: Crawl4AIResult[] | Record<string, Crawl4AIResult>;
  crawled?: Crawl4AIResult[] | Record<string, Crawl4AIResult>;
};

const normalizeCrawl4AIResults = (
  payload: Crawl4AIResponse | Crawl4AIResult[],
) => {
  if (!payload) {
    return [];
  }

  if (Array.isArray(payload)) {
    return payload;
  }

  const container =
    payload.results ?? payload.data ?? payload.items ?? payload.crawled ?? payload;

  if (Array.isArray(container)) {
    return container;
  }

  if (container && typeof container === 'object') {
    const entries = Object.entries(container);
    const looksLikeUrlMap = entries.some(([url]) =>
      url.startsWith('http://') || url.startsWith('https://'),
    );

    if (!looksLikeUrlMap) {
      return [];
    }

    return entries.map(([url, value]) => {
      const safeValue =
        value && typeof value === 'object' ? (value as Crawl4AIResult) : {};

      return {
        ...safeValue,
        url: safeValue.url ?? url,
      };
    });
  }

  return [];
};

const resolveResultUrl = (result: Crawl4AIResult, fallbackUrl: string) =>
  result.url ??
  result.source_url ??
  result.original_url ??
  result.final_url ??
  fallbackUrl;

const resolveResultTitle = (result: Crawl4AIResult, fallbackUrl: string) =>
  result.title ??
  result.page_title ??
  result.metadata?.title ??
  result.meta?.title ??
  `Content from ${fallbackUrl}`;

const resolveResultContent = (result: Crawl4AIResult) => {
  if (result.markdown) return result.markdown;
  if (result.content) return result.content;
  if (result.text) return result.text;
  if (result.extracted_text) return result.extracted_text;

  const html = result.cleaned_html ?? result.html;
  return html ? turndownService.turndown(html) : '';
};

const schema = z.object({
  urls: z.array(z.string()).describe('A list of URLs to scrape content from.'),
});

const actionDescription = `
Use this tool to scrape and extract content from the provided URLs. This is useful when you the user has asked you to extract or summarize information from specific web pages. You can provide up to 3 URLs at a time. NEVER CALL THIS TOOL EXPLICITLY YOURSELF UNLESS INSTRUCTED TO DO SO BY THE USER.
You should only call this tool when the user has specifically requested information from certain web pages, never call this yourself to get extra information without user instruction.

For example, if the user says "Please summarize the content of https://example.com/article", you can call this tool with that URL to get the content and then provide the summary or "What does X mean according to https://example.com/page", you can call this tool with that URL to get the content and provide the explanation.
`;

const scrapeURLAction: ResearchAction<typeof schema> = {
  name: 'scrape_url',
  schema: schema,
  getToolDescription: () =>
    'Use this tool to scrape and extract content from the provided URLs. This is useful when you the user has asked you to extract or summarize information from specific web pages. You can provide up to 3 URLs at a time. NEVER CALL THIS TOOL EXPLICITLY YOURSELF UNLESS INSTRUCTED TO DO SO BY THE USER.',
  getDescription: () => actionDescription,
  enabled: (_) => true,
  execute: async (params, additionalConfig) => {
    const urls = params.urls.slice(0, 3);

    let readingBlockId = crypto.randomUUID();
    let readingEmitted = false;

    const researchBlockId = additionalConfig.researchBlockId;
    const researchBlock = additionalConfig.session.getBlock(researchBlockId);

    const results: Chunk[] = [];

    let crawlResults: Crawl4AIResult[] = [];

    try {
      const crawl4aiEndpoints = getCrawl4aiURLs();
      const endpoint =
        crawl4aiEndpoints.length > 0
          ? pickCrawl4aiEndpoint(crawl4aiEndpoints)
          : 'http://localhost:11235/crawl-lite';
      const payload = {
        urls,
        browser_config: {
          type: 'BrowserConfig',
          params: { headless: true },
        },
        crawler_config: {
          type: 'CrawlerRunConfig',
          params: { cache_mode: 'bypass' },
        },
      };

      const response = await fetchCrawl4AiWithRetry(endpoint, payload);

      const responseText = await response.text();
      let responseJson: Crawl4AIResponse | Crawl4AIResult[] = [];

      try {
        responseJson = JSON.parse(responseText) as
          | Crawl4AIResponse
          | Crawl4AIResult[];
      } catch (error) {
        throw new Error(
          `crawl4ai returned non-JSON response: ${responseText.slice(0, 200)}`,
        );
      }

      crawlResults = normalizeCrawl4AIResults(responseJson);
    } catch (error) {
      urls.forEach((url) => {
        results.push({
          content: `Failed to fetch content from ${url}: ${error}`,
          metadata: {
            url,
            title: `Error fetching ${url}`,
            source: 'scrape_url',
            scraped: true,
            scrapeError: true,
          },
        });
      });

      return {
        type: 'search_results',
        results,
      };
    }

    const crawlResultsByUrl = new Map<string, Crawl4AIResult>();
    crawlResults.forEach((result) => {
      const candidates = [
        result.url,
        result.source_url,
        result.original_url,
        result.final_url,
      ].filter(Boolean) as string[];

      candidates.forEach((candidate) => {
        crawlResultsByUrl.set(candidate, result);
      });
    });

    urls.forEach((url, index) => {
      const crawlResult = crawlResultsByUrl.get(url) ?? crawlResults[index];

      if (!crawlResult) {
        results.push({
          content: `Failed to fetch content from ${url}: crawl4ai returned no result`,
          metadata: {
            url,
            title: `Error fetching ${url}`,
            source: 'scrape_url',
            scraped: true,
            scrapeError: true,
          },
        });
        return;
      }

      const resolvedUrl = resolveResultUrl(crawlResult, url);
      const title = resolveResultTitle(crawlResult, resolvedUrl);
      const content = resolveResultContent(crawlResult);
      const hasContent = Boolean(content);
      const finalContent =
        content ||
        `Failed to extract content from ${resolvedUrl}: empty response`;

      if (researchBlock && researchBlock.type === 'research') {
        if (!readingEmitted) {
          readingEmitted = true;
          additionalConfig.session.updateBlock(researchBlockId, [
            {
              op: 'add',
              path: '/data/subSteps/-',
              value: {
                id: readingBlockId,
                type: 'reading',
                reading: [
                  {
                    content: '',
                    metadata: {
                      url: resolvedUrl,
                      title: title,
                    },
                  },
                ],
              },
            },
          ]);
        } else {
          const latestBlock = additionalConfig.session.getBlock(researchBlockId);
          if (latestBlock && latestBlock.type === 'research') {
            const subStepIndex = latestBlock.data.subSteps.findIndex(
              (step: any) => step.id === readingBlockId,
            );

            if (subStepIndex === -1) {
              additionalConfig.session.updateBlock(researchBlockId, [
                {
                  op: 'add',
                  path: '/data/subSteps/-',
                  value: {
                    id: readingBlockId,
                    type: 'reading',
                    reading: [
                      {
                        content: '',
                        metadata: {
                          url: resolvedUrl,
                          title: title,
                        },
                      },
                    ],
                  },
                },
              ]);
            } else {
              additionalConfig.session.updateBlock(researchBlockId, [
                {
                  op: 'add',
                  path: `/data/subSteps/${subStepIndex}/reading/-`,
                  value: {
                    content: '',
                    metadata: {
                      url: resolvedUrl,
                      title: title,
                    },
                  },
                },
              ]);
            }
          }
        }
      }

      results.push({
        content: finalContent,
        metadata: {
          url: resolvedUrl,
          title: title,
          source: 'scrape_url',
          scraped: true,
          ...(hasContent ? {} : { scrapeError: true }),
        },
      });
    });

    return {
      type: 'search_results',
      results,
    };
  },
};

export default scrapeURLAction;

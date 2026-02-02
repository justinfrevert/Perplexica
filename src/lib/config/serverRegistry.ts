import configManager from './index';
import { ConfigModelProvider } from './types';

const DEFAULT_CRAWL4AI_ENDPOINT = 'http://localhost:11235/crawl-lite';

export const getConfiguredModelProviders = (): ConfigModelProvider[] => {
  return configManager.getConfig('modelProviders', []);
};

export const getConfiguredModelProviderById = (
  id: string,
): ConfigModelProvider | undefined => {
  return getConfiguredModelProviders().find((p) => p.id === id) ?? undefined;
};

export const getSearxngURLs = (): string[] => {
  configManager.refreshFromDisk();
  const configured = configManager.getConfig('search.searxngURLs', []);
  const urlList = Array.isArray(configured)
    ? configured
        .filter((url) => typeof url === 'string')
        .map((url) => url.trim())
        .filter(Boolean)
    : [];

  if (urlList.length > 0) {
    return urlList;
  }

  const single = configManager.getConfig('search.searxngURL', '');
  if (typeof single === 'string' && single.trim()) {
    return [single.trim()];
  }

  return [];
};

export const getSearxngURL = () => getSearxngURLs()[0] ?? '';

export const getCrawl4aiURLs = (): string[] => {
  configManager.refreshFromDisk();
  const configured = configManager.getConfig('search.crawl4aiURLs', []);
  const urlList = Array.isArray(configured)
    ? configured
        .filter((url) => typeof url === 'string')
        .map((url) => url.trim())
        .filter(Boolean)
    : [];

  if (urlList.length > 0) {
    return urlList;
  }

  const single = configManager.getConfig('search.crawl4aiURL', '');
  if (typeof single === 'string' && single.trim()) {
    return [single.trim()];
  }

  const envEndpoint = process.env.CRAWL4AI_ENDPOINT;
  if (typeof envEndpoint === 'string' && envEndpoint.trim()) {
    return [envEndpoint.trim()];
  }

  return [DEFAULT_CRAWL4AI_ENDPOINT];
};

export const getCrawl4aiURL = () =>
  getCrawl4aiURLs()[0] ?? DEFAULT_CRAWL4AI_ENDPOINT;

import configManager from './index';
import { ConfigModelProvider } from './types';

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

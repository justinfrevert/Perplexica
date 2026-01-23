import axios from 'axios';
import { getSearxngURLs } from './config/serverRegistry';

interface SearxngSearchOptions {
  categories?: string[];
  engines?: string[];
  language?: string;
  pageno?: number;
}

interface SearxngSearchResult {
  title: string;
  url: string;
  img_src?: string;
  thumbnail_src?: string;
  thumbnail?: string;
  content?: string;
  author?: string;
  iframe_src?: string;
}

export const searchSearxng = async (
  query: string,
  opts?: SearxngSearchOptions,
) => {
  const searxngURLs = getSearxngURLs();
  if (searxngURLs.length === 0) {
    throw new Error('No SearXNG URLs are configured.');
  }

  const searchParams = new URLSearchParams();
  searchParams.set('format', 'json');
  searchParams.append('q', query);

  if (opts) {
    Object.keys(opts).forEach((key) => {
      const value = opts[key as keyof SearxngSearchOptions];
      if (Array.isArray(value)) {
        searchParams.append(key, value.join(','));
        return;
      }
      searchParams.append(key, String(value));
    });
  }

  const baseURL = pickSearxngURL(searxngURLs, searchParams.toString());
  const url = new URL(`${baseURL}/search`);
  url.search = searchParams.toString();

  console.info(`[searxng] using instance: ${baseURL}`);

  const res = await fetch(url);
  const data = await res.json();

  const results: SearxngSearchResult[] = data.results;
  const suggestions: string[] = data.suggestions;

  return { results, suggestions };
};

const pickSearxngURL = (urls: string[], key: string) => {
  if (urls.length === 1) {
    return urls[0];
  }

  const hash = hashString(key);
  const index = hash % urls.length;

  return urls[index];
};

// Deterministic hash so the same query maps to the same instance.
const hashString = (value: string) => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }

  return hash >>> 0;
};

export type SearchDebugLogger = (...args: unknown[]) => void;

const truthyValues = new Set(['1', 'true', 'yes', 'on']);

export const isSearchDebugEnabled = () => {
  const flag = (process.env.DEBUG_SEARCH ?? '').toLowerCase().trim();
  return truthyValues.has(flag);
};

export const createSearchDebugLogger = (
  sessionId?: string | null,
  enabled?: boolean,
): SearchDebugLogger => {
  const active = enabled ?? isSearchDebugEnabled();
  const prefix = sessionId ? `[search:${sessionId}]` : '[search]';

  return (...args: unknown[]) => {
    if (!active) return;
    console.log(prefix, ...args);
  };
};

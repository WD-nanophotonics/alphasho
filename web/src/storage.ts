// Keep production's existing saves; an independent preview must not overwrite
// games or preferences belonging to another Pages application on this origin.
export function storageKey(key: string, base = import.meta.env.BASE_URL): string {
  return base === '/alphasho/' ? key : key + ':' + base;
}

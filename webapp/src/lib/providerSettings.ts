import type { ProviderCredentials } from './types'

// Provider credentials never leave this browser except as part of a
// POST /api/runs body the user themselves triggers (see NewRun.tsx) — the
// backend does not persist them (see codoctopus/server/runs.py). This is a
// deliberate trade-off: it keeps a local, no-auth dev tool from ever writing
// secrets to a shared server; the cost is re-entering them in a new browser
// or after clearing site data.
const STORAGE_KEY = 'codoctopus.providerCredentials'

type Store = Record<string, ProviderCredentials>

function readStore(): Store {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as Store) : {}
  } catch {
    return {}
  }
}

function writeStore(store: Store): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store))
  } catch {
    // localStorage unavailable (private browsing, quota) — settings just won't persist
  }
}

export function getAllProviderCredentials(): Store {
  return readStore()
}

export function getProviderCredentials(provider: string): ProviderCredentials {
  return readStore()[provider] ?? {}
}

export function setProviderCredentials(provider: string, creds: ProviderCredentials): void {
  const store = readStore()
  const trimmed: ProviderCredentials = {}
  if (creds.api_key?.trim()) trimmed.api_key = creds.api_key.trim()
  if (creds.base_url?.trim()) trimmed.base_url = creds.base_url.trim()

  if (Object.keys(trimmed).length === 0) {
    delete store[provider]
  } else {
    store[provider] = trimmed
  }
  writeStore(store)
}

/** "anthropic:claude-opus-5" -> "anthropic"; a bare or missing ref falls back to "anthropic" (the server's own default provider). */
export function providerFromModelRef(ref: string | null | undefined): string {
  if (!ref) return 'anthropic'
  const [provider] = ref.split(':')
  return provider || 'anthropic'
}

/**
 * Which provider a fresh run form should preselect: the first one (in
 * server-reported order) that actually has a saved API key, so a form that
 * defaults to "anthropic" can't silently ignore a Gemini/OpenAI key someone
 * configured in Settings and only that one. Falls back to "anthropic" (the
 * server's own hardcoded default) when nothing is configured yet.
 */
export function pickDefaultProvider(providers: string[]): string {
  const store = readStore()
  const configured = providers.find((p) => store[p]?.api_key)
  if (configured) return configured
  return providers.includes('anthropic') ? 'anthropic' : (providers[0] ?? 'anthropic')
}

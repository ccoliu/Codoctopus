import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { api } from '../lib/apiClient'
import { getProviderCredentials, setProviderCredentials } from '../lib/providerSettings'
import type { ProviderCredentials } from '../lib/types'

const inputClass =
  'w-full rounded-lg border border-border bg-plane px-3 py-2 text-sm text-ink outline-none focus:border-accent'

function ProviderRow({ provider }: { provider: string }) {
  const [creds, setCreds] = useState<ProviderCredentials>(() => getProviderCredentials(provider))
  const [saved, setSaved] = useState(false)

  useEffect(() => setSaved(false), [creds])

  // "gateway" is a separate provider from "openai" precisely so a local
  // LM Studio/vLLM/etc. server isn't labeled as if it were the real OpenAI
  // API — it always needs a base_url, unlike "openai" which never does.
  const isGateway = provider === 'gateway'

  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-sm font-medium text-ink">{provider}</span>
        {saved && <span className="text-xs text-status-good">Saved</span>}
      </div>
      {isGateway && (
        <p className="mb-3 text-xs text-ink-secondary">
          For an OpenAI-compatible local server (LM Studio, vLLM, ...) — not the real OpenAI API.
        </p>
      )}
      <div className="flex flex-col gap-3">
        {!isGateway && (
          <div>
            <label className="mb-1.5 block text-xs font-medium text-ink-secondary">API key</label>
            <input
              type="password"
              value={creds.api_key ?? ''}
              onChange={(e) => setCreds((c) => ({ ...c, api_key: e.target.value }))}
              placeholder={`${provider.toUpperCase()}_API_KEY`}
              autoComplete="off"
              className={inputClass}
            />
          </div>
        )}
        {isGateway && (
          <>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-ink-secondary">
                Base URL <span className="font-normal text-ink-muted">(required)</span>
              </label>
              <input
                value={creds.base_url ?? ''}
                onChange={(e) => setCreds((c) => ({ ...c, base_url: e.target.value }))}
                placeholder="http://localhost:1234/v1"
                className={inputClass}
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-ink-secondary">
                API key <span className="font-normal text-ink-muted">(only if your server checks one)</span>
              </label>
              <input
                type="password"
                value={creds.api_key ?? ''}
                onChange={(e) => setCreds((c) => ({ ...c, api_key: e.target.value }))}
                placeholder="usually not needed"
                autoComplete="off"
                className={inputClass}
              />
            </div>
          </>
        )}
        <button
          onClick={() => {
            setProviderCredentials(provider, creds)
            setSaved(true)
          }}
          className="self-start rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-strong"
        >
          Save
        </button>
      </div>
    </Card>
  )
}

export function Settings() {
  const providersQuery = useQuery({ queryKey: ['providers'], queryFn: api.listProviders })

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="mb-1 text-xl font-semibold text-ink">Settings</h1>
      <p className="mb-6 text-sm text-ink-secondary">
        Credentials are saved only in this browser (localStorage) and sent with each run you start —
        the server never stores them. Ollama needs none of this; it talks to a local server by default.
      </p>

      {providersQuery.isLoading && (
        <div className="flex items-center gap-2 text-sm text-ink-secondary">
          <Spinner /> Loading providers…
        </div>
      )}

      <div className="flex flex-col gap-3">
        {providersQuery.data?.providers
          .filter((p) => p !== 'ollama')
          .map((p) => <ProviderRow key={p} provider={p} />)}
      </div>
    </div>
  )
}

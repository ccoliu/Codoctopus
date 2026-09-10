import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { api } from '../lib/apiClient'
import { getProviderCredentials, pickDefaultProvider } from '../lib/providerSettings'

const inputClass =
  'w-full rounded-lg border border-border bg-plane px-3 py-2 text-sm text-ink outline-none focus:border-accent'
const labelClass = 'mb-1.5 block text-sm font-medium text-ink-secondary'

function modelRef(provider: string, modelName: string): string | null {
  if (!provider) return null
  const trimmed = modelName.trim()
  return trimmed ? `${provider}:${trimmed}` : provider
}

/**
 * A dropdown of the provider's actual available models when they can be
 * fetched (Ollama needs no key; others need one already saved in Settings),
 * falling back to a free-text input otherwise — an unlisted custom provider,
 * a fetch error, or credentials not saved yet all degrade to typing it in.
 */
function ModelNameField({
  label,
  provider,
  value,
  onChange,
}: {
  label: string
  provider: string
  value: string
  onChange: (v: string) => void
}) {
  const creds = getProviderCredentials(provider)
  const modelsQuery = useQuery({
    queryKey: ['models', provider, creds.api_key, creds.base_url],
    queryFn: () => api.listProviderModels(provider, creds),
    enabled: !!provider,
    retry: false,
    staleTime: 60_000,
  })
  const models = modelsQuery.data?.models ?? []

  if (models.length > 0) {
    return (
      <div>
        <label className={labelClass}>{label}</label>
        <select value={value} onChange={(e) => onChange(e.target.value)} className={`${inputClass} bg-surface`}>
          <option value="">({provider}&apos;s default model)</option>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
    )
  }

  return (
    <div>
      <label className={labelClass}>{label}</label>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={
          modelsQuery.isFetching
            ? 'Loading models…'
            : modelsQuery.isError
              ? `Could not fetch models — type a name`
              : `${provider || 'provider'}'s default model`
        }
        className={`${inputClass} bg-surface`}
      />
    </div>
  )
}

export function NewRun() {
  const navigate = useNavigate()
  const [goal, setGoal] = useState('')
  const [domain, setDomain] = useState('')
  const [executor, setExecutor] = useState<'local' | 'coworkify'>('local')
  const [showAdvanced, setShowAdvanced] = useState(false)

  const [plannerProvider, setPlannerProvider] = useState('')
  const [plannerModelName, setPlannerModelName] = useState('')

  // Off by default: the worker step then just uses whatever the planner is
  // set to, so switching the one "Provider" select is enough to point both
  // at the same model — you only need to touch this if they should diverge.
  const [splitWorkerModel, setSplitWorkerModel] = useState(false)
  const [workerProvider, setWorkerProvider] = useState('')
  const [workerModelName, setWorkerModelName] = useState('')

  const domainsQuery = useQuery({ queryKey: ['domains'], queryFn: api.listDomains })
  const providersQuery = useQuery({ queryKey: ['providers'], queryFn: api.listProviders })
  const providers = providersQuery.data?.providers ?? []

  // Preselect whichever provider actually has a saved key (Settings), so this
  // never silently falls back to a provider the user hasn't configured — that
  // was the bug: leaving these blank meant "anthropic" no matter what key
  // you'd actually saved. Only seeds an empty selection, never overwrites one
  // the user already made.
  useEffect(() => {
    if (providers.length === 0) return
    const preferred = pickDefaultProvider(providers)
    setPlannerProvider((p) => p || preferred)
    setWorkerProvider((p) => p || preferred)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providersQuery.data])

  const effectiveWorkerProvider = splitWorkerModel ? workerProvider : plannerProvider
  const effectiveWorkerModelName = splitWorkerModel ? workerModelName : plannerModelName

  const createRun = useMutation({
    mutationFn: () => {
      const plannerCreds = getProviderCredentials(plannerProvider)
      const workerCreds = getProviderCredentials(effectiveWorkerProvider)
      return api.createRun({
        goal,
        domain: domain || null,
        executor,
        model: modelRef(plannerProvider, plannerModelName),
        worker_model: modelRef(effectiveWorkerProvider, effectiveWorkerModelName),
        planner_api_key: plannerCreds.api_key ?? null,
        planner_base_url: plannerCreds.base_url ?? null,
        worker_api_key: workerCreds.api_key ?? null,
        worker_base_url: workerCreds.base_url ?? null,
      })
    },
    onSuccess: (run) => navigate(`/runs/${run.id}`),
  })

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="mb-1 text-xl font-semibold text-ink">New run</h1>
      <p className="mb-6 text-sm text-ink-secondary">
        Describe a goal. Codoctopus plans it into steps, then runs them and streams progress here. No
        API key saved yet?{' '}
        <Link to="/settings" className="text-accent hover:text-accent-strong">
          Add one in Settings
        </Link>
        .
      </p>

      <Card className="p-6">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            if (goal.trim() && !createRun.isPending) createRun.mutate()
          }}
          className="flex flex-col gap-4"
        >
          <div>
            <label className={labelClass}>Goal</label>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={3}
              required
              placeholder="e.g. Write a hello world script and save it as hello.py"
              className={inputClass}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={labelClass}>Domain</label>
              <select value={domain} onChange={(e) => setDomain(e.target.value)} className={inputClass}>
                <option value="">(none — general)</option>
                {domainsQuery.data?.domains.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className={labelClass}>Executor</label>
              <select
                value={executor}
                onChange={(e) => setExecutor(e.target.value as 'local' | 'coworkify')}
                className={inputClass}
              >
                <option value="local">local</option>
                <option value="coworkify">coworkify</option>
              </select>
            </div>
          </div>

          <div>
            <label className={labelClass}>Provider</label>
            <select
              value={plannerProvider}
              onChange={(e) => setPlannerProvider(e.target.value)}
              className={inputClass}
            >
              {providers.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <label className="flex items-center gap-2 text-sm text-ink-secondary">
            <input
              type="checkbox"
              checked={splitWorkerModel}
              onChange={(e) => setSplitWorkerModel(e.target.checked)}
              className="h-4 w-4 rounded border-border accent-accent"
            />
            Use a different model for the worker step
          </label>

          {splitWorkerModel && (
            <div>
              <label className={labelClass}>Worker provider</label>
              <select
                value={workerProvider}
                onChange={(e) => setWorkerProvider(e.target.value)}
                className={inputClass}
              >
                {providers.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="self-start text-sm font-medium text-accent hover:text-accent-strong"
          >
            {showAdvanced ? 'Hide' : 'Show'} advanced options
          </button>

          {showAdvanced && (
            <div
              className={`grid gap-4 rounded-lg border border-border bg-plane p-4 ${splitWorkerModel ? 'grid-cols-2' : 'grid-cols-1'}`}
            >
              <ModelNameField
                label={splitWorkerModel ? 'Planner model name' : 'Model name'}
                provider={plannerProvider}
                value={plannerModelName}
                onChange={setPlannerModelName}
              />
              {splitWorkerModel && (
                <ModelNameField
                  label="Worker model name"
                  provider={workerProvider}
                  value={workerModelName}
                  onChange={setWorkerModelName}
                />
              )}
            </div>
          )}

          {createRun.isError && (
            <p className="text-sm text-status-critical">{(createRun.error as Error).message}</p>
          )}

          <button
            type="submit"
            disabled={createRun.isPending || !goal.trim()}
            className="inline-flex items-center justify-center gap-2 self-start rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
          >
            {createRun.isPending && <Spinner className="h-4 w-4 text-white" />}
            Run
          </button>
        </form>
      </Card>
    </div>
  )
}

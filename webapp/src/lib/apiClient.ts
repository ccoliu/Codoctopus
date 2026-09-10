import type {
  ProviderCredentials,
  RunCreateBody,
  RunDetail,
  RunSummary,
  ScheduleCreateBody,
  ScheduleResponse,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listDomains: () => request<{ domains: string[] }>('/domains'),
  listProviders: () => request<{ providers: string[] }>('/providers'),
  listRuns: () => request<{ runs: RunSummary[] }>('/runs'),
  getRun: (id: string) => request<RunDetail>(`/runs/${id}`),
  createRun: (body: RunCreateBody) =>
    request<RunDetail>('/runs', { method: 'POST', body: JSON.stringify(body) }),
  listProviderModels: (provider: string, creds: ProviderCredentials) =>
    request<{ models: string[] }>(`/providers/${provider}/models`, {
      method: 'POST',
      body: JSON.stringify(creds),
    }),
  createSchedule: (runId: string, body: ScheduleCreateBody) =>
    request<ScheduleResponse>(`/runs/${runId}/schedule`, { method: 'POST', body: JSON.stringify(body) }),
}

export function runStreamUrl(runId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/api/runs/${runId}/stream`
}

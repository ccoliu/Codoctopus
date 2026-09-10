export type RunStatus = 'planning' | 'running' | 'success' | 'failed'

export interface PlanStep {
  key: string
  name: string
  role: string
  instruction: string
  depends_on: string[]
  tools: string[]
  for_each: string | null
}

export interface Plan {
  goal: string
  domain: string
  steps: PlanStep[]
}

export interface RunSummary {
  id: string
  goal: string
  domain: string | null
  executor: string
  status: RunStatus
  created_at: number
}

export interface RunDetail extends RunSummary {
  plan: Plan | null
  step_results: Record<string, string>
  error: string | null
}

export interface RunCreateBody {
  goal: string
  domain?: string | null
  model?: string | null
  worker_model?: string | null
  executor?: 'local' | 'coworkify'
  workspace?: string | null
  planner_api_key?: string | null
  planner_base_url?: string | null
  worker_api_key?: string | null
  worker_base_url?: string | null
}

/** Credentials for one provider, held only in the browser (see providerSettings.ts). */
export interface ProviderCredentials {
  api_key?: string
  base_url?: string
}

export type StepRunStatus = 'pending' | 'running' | 'done' | 'failed'

export interface StepRunState {
  key: string
  status: StepRunStatus
  result?: string
  error?: string
}

export type StreamEvent =
  | { event: 'snapshot'; data: RunDetail }
  | { event: 'plan_ready'; data: { plan: Plan } }
  | { event: 'step_started'; data: { key: string } }
  | { event: 'step_done'; data: { key: string; result: string } }
  | { event: 'step_failed'; data: { key: string; error: string } }
  | {
      event: 'run_done'
      data: { status: RunStatus; step_results: Record<string, string>; error: string | null }
    }

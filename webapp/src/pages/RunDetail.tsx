import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { StatusBadge } from '../components/StatusBadge'
import { useRunStream } from '../hooks/useRunStream'
import { api } from '../lib/apiClient'
import type { PlanStep, StepRunState } from '../lib/types'

const inputClass =
  'w-full rounded-lg border border-border bg-plane px-3 py-2 text-sm text-ink outline-none focus:border-accent'
const labelClass = 'mb-1.5 block text-sm font-medium text-ink-secondary'

/**
 * Registers a run's already-planned steps as a recurring Coworkify cron
 * schedule — a one-way "create" action. Editing/disabling/deleting a
 * schedule afterward stays in Coworkify's own dashboard rather than being
 * duplicated here.
 */
function ScheduleForm({ runId, goal }: { runId: string; goal: string }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState(goal.slice(0, 60))
  const [cron, setCron] = useState('0 9 * * *')

  const createSchedule = useMutation({
    mutationFn: () => api.createSchedule(runId, { name, cron_expression: cron }),
  })

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="text-sm font-medium text-accent hover:text-accent-strong"
      >
        Schedule as cron…
      </button>
    )
  }

  if (createSchedule.isSuccess) {
    const sched = createSchedule.data
    return (
      <Card className="p-4 text-sm">
        <p className="text-ink">
          Scheduled <span className="font-medium">{sched.name}</span> ({sched.cron_expression}).
        </p>
        {sched.next_run_at && <p className="mt-1 text-ink-secondary">Next run: {sched.next_run_at}</p>}
        <p className="mt-1 text-ink-secondary">
          Manage it (edit, disable, delete) from Coworkify&apos;s own Schedules page.
        </p>
      </Card>
    )
  }

  return (
    <Card className="p-4">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (name.trim() && cron.trim() && !createSchedule.isPending) createSchedule.mutate()
        }}
        className="flex flex-col gap-3"
      >
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelClass}>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} className={inputClass} required />
          </div>
          <div>
            <label className={labelClass}>Cron expression</label>
            <input
              value={cron}
              onChange={(e) => setCron(e.target.value)}
              placeholder="0 9 * * *"
              className={`${inputClass} font-mono`}
              required
            />
          </div>
        </div>
        {createSchedule.isError && (
          <p className="text-sm text-status-critical">{(createSchedule.error as Error).message}</p>
        )}
        <div className="flex gap-2">
          <button
            type="submit"
            disabled={createSchedule.isPending}
            className="inline-flex items-center gap-2 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
          >
            {createSchedule.isPending && <Spinner className="h-4 w-4 text-white" />}
            Create schedule
          </button>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="rounded-lg px-3 py-1.5 text-sm text-ink-secondary hover:text-ink"
          >
            Cancel
          </button>
        </div>
      </form>
    </Card>
  )
}

function itemsFor(step: PlanStep, steps: Record<string, StepRunState>): StepRunState[] {
  if (!step.for_each) {
    const s = steps[step.key]
    return s ? [s] : []
  }
  return Object.values(steps)
    .filter((s) => s.key.startsWith(`${step.key}[`))
    .sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }))
}

function overallStatus(items: StepRunState[]): string {
  if (items.length === 0) return 'pending'
  if (items.some((s) => s.status === 'failed')) return 'failed'
  if (items.every((s) => s.status === 'done')) return 'done'
  return 'running'
}

function StepCard({ step, steps }: { step: PlanStep; steps: Record<string, StepRunState> }) {
  const items = itemsFor(step, steps)
  const overall = overallStatus(items)

  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs text-ink-muted">{step.key}</span>
            <span className="text-sm font-medium text-ink">{step.name}</span>
            {step.for_each && (
              <span className="rounded-full bg-plane px-2 py-0.5 text-xs text-ink-muted">
                for_each: {step.for_each}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-ink-secondary">{step.instruction}</p>
          {step.depends_on.length > 0 && (
            <p className="mt-1 text-xs text-ink-muted">depends on: {step.depends_on.join(', ')}</p>
          )}
        </div>
        <StatusBadge status={overall} />
      </div>

      {!step.for_each && items[0]?.status === 'done' && (
        <pre className="mt-3 overflow-x-auto whitespace-pre-wrap rounded-lg bg-plane p-3 text-xs text-ink-secondary">
          {items[0].result}
        </pre>
      )}
      {!step.for_each && items[0]?.status === 'failed' && (
        <p className="mt-3 text-xs text-status-critical">{items[0].error}</p>
      )}

      {step.for_each && items.length > 0 && (
        <div className="mt-3 flex flex-col gap-3 border-t border-border pt-3">
          {items.map((item) => (
            <div key={item.key}>
              <div className="mb-1 flex items-center gap-2">
                <span className="font-mono text-xs text-ink-muted">{item.key}</span>
                <StatusBadge status={item.status} />
              </div>
              {item.status === 'done' && (
                <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-plane p-3 text-xs text-ink-secondary">
                  {item.result}
                </pre>
              )}
              {item.status === 'failed' && <p className="text-xs text-status-critical">{item.error}</p>}
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

export function RunDetail() {
  const { id } = useParams<{ id: string }>()
  const { run, steps, connected } = useRunStream(id)

  if (!run) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-secondary">
        <Spinner /> Loading run…
      </div>
    )
  }

  const finished = run.status === 'success' || run.status === 'failed'

  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-ink">{run.goal}</h1>
          <p className="mt-1 text-sm text-ink-secondary">
            {run.domain ?? 'general'} &middot; {run.executor}
            {!connected && !finished && <span className="ml-2 text-status-warning">(reconnecting…)</span>}
          </p>
          {run.workspace && (
            <p className="mt-1 truncate font-mono text-xs text-ink-muted" title={run.workspace}>
              {run.workspace}
            </p>
          )}
        </div>
        <StatusBadge status={run.status} />
      </div>

      {run.error && (
        <Card className="mb-6 border-status-critical/30 bg-status-critical/5 p-4">
          <p className="text-sm text-status-critical">{run.error}</p>
        </Card>
      )}

      {!run.plan ? (
        <div className="flex items-center gap-2 text-sm text-ink-secondary">
          <Spinner /> Planning…
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {run.plan.steps.map((step) => (
            <StepCard key={step.key} step={step} steps={steps} />
          ))}
          <div className="mt-2">
            <ScheduleForm runId={run.id} goal={run.goal} />
          </div>
        </div>
      )}
    </div>
  )
}

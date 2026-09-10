import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { StatusBadge } from '../components/StatusBadge'
import { api } from '../lib/apiClient'

export function RunsList() {
  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: api.listRuns,
    refetchInterval: 3000,
  })

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-6 text-xl font-semibold text-ink">Runs</h1>

      {runsQuery.isLoading && (
        <div className="flex items-center gap-2 text-sm text-ink-secondary">
          <Spinner /> Loading…
        </div>
      )}

      {runsQuery.data?.runs.length === 0 && (
        <p className="text-sm text-ink-secondary">
          No runs yet. <Link to="/" className="text-accent hover:text-accent-strong">Start one</Link>.
        </p>
      )}

      <div className="flex flex-col gap-2">
        {runsQuery.data?.runs.map((run) => (
          <Link key={run.id} to={`/runs/${run.id}`}>
            <Card className="flex items-center justify-between gap-4 p-4 transition-colors hover:border-accent/40">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-ink">{run.goal}</p>
                <p className="mt-0.5 text-xs text-ink-muted">
                  {run.domain ?? 'general'} &middot; {run.executor} &middot;{' '}
                  {new Date(run.created_at * 1000).toLocaleString()}
                </p>
              </div>
              <StatusBadge status={run.status} />
            </Card>
          </Link>
        ))}
      </div>
    </div>
  )
}

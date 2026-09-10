type Status = 'planning' | 'pending' | 'running' | 'success' | 'done' | 'failed'

const STYLES: Record<Status, { dot: string; text: string; label: string }> = {
  planning: { dot: 'bg-ink-muted animate-pulse', text: 'text-ink-secondary', label: 'Planning' },
  pending: { dot: 'bg-ink-muted', text: 'text-ink-secondary', label: 'Pending' },
  running: { dot: 'bg-accent animate-pulse', text: 'text-accent', label: 'Running' },
  success: { dot: 'bg-status-good', text: 'text-status-good', label: 'Success' },
  done: { dot: 'bg-status-good', text: 'text-status-good', label: 'Done' },
  failed: { dot: 'bg-status-critical', text: 'text-status-critical', label: 'Failed' },
}

export function StatusBadge({ status }: { status: string }) {
  const style = STYLES[status as Status] ?? STYLES.pending
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${style.text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  )
}

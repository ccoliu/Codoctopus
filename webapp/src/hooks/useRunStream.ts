import { useEffect, useRef, useState } from 'react'
import { runStreamUrl } from '../lib/apiClient'
import type { RunDetail, StepRunState, StreamEvent } from '../lib/types'

interface RunStreamState {
  run: RunDetail | null
  steps: Record<string, StepRunState>
  connected: boolean
}

/** Subscribes to a run's WebSocket stream and keeps live run + per-step state. */
export function useRunStream(runId: string | undefined): RunStreamState {
  const [run, setRun] = useState<RunDetail | null>(null)
  const [steps, setSteps] = useState<Record<string, StepRunState>>({})
  const [connected, setConnected] = useState(false)
  const runIdRef = useRef(runId)
  runIdRef.current = runId

  useEffect(() => {
    if (!runId) return
    setRun(null)
    setSteps({})

    const ws = new WebSocket(runStreamUrl(runId))

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)

    ws.onmessage = (raw) => {
      const msg = JSON.parse(raw.data) as StreamEvent
      switch (msg.event) {
        case 'snapshot': {
          setRun(msg.data)
          const seeded: Record<string, StepRunState> = {}
          for (const [key, result] of Object.entries(msg.data.step_results)) {
            seeded[key] = { key, status: 'done', result }
          }
          setSteps((prev) => ({ ...seeded, ...prev }))
          break
        }
        case 'plan_ready':
          setRun((prev) =>
            prev ? { ...prev, plan: msg.data.plan, workspace: msg.data.workspace, status: 'running' } : prev,
          )
          break
        case 'step_started':
          setSteps((prev) => ({ ...prev, [msg.data.key]: { key: msg.data.key, status: 'running' } }))
          break
        case 'step_done':
          setSteps((prev) => ({
            ...prev,
            [msg.data.key]: { key: msg.data.key, status: 'done', result: msg.data.result },
          }))
          break
        case 'step_failed':
          setSteps((prev) => ({
            ...prev,
            [msg.data.key]: { key: msg.data.key, status: 'failed', error: msg.data.error },
          }))
          break
        case 'run_done':
          setRun((prev) =>
            prev
              ? {
                  ...prev,
                  status: msg.data.status,
                  step_results: msg.data.step_results,
                  error: msg.data.error,
                }
              : prev,
          )
          break
      }
    }

    return () => ws.close()
  }, [runId])

  return { run, steps, connected }
}

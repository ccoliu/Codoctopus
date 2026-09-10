import clsx from 'clsx'
import { NavLink, Outlet } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'New Run', end: true },
  { to: '/runs', label: 'Runs', end: false },
  { to: '/settings', label: 'Settings', end: false },
]

export function Layout() {
  return (
    <div className="min-h-screen bg-plane">
      <div className="mx-auto flex min-h-screen max-w-5xl">
        <aside className="hidden w-56 shrink-0 flex-col border-r border-border px-4 py-6 sm:flex">
          <div className="mb-8 flex items-center gap-2 px-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-sm font-bold text-white">
              C
            </span>
            <span className="text-sm font-semibold text-ink">Codoctopus</span>
          </div>
          <nav className="flex flex-col gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    'rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-accent/10 text-accent'
                      : 'text-ink-secondary hover:bg-surface hover:text-ink',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-center justify-between border-b border-border px-6 py-4 sm:hidden">
            <span className="text-sm font-semibold text-ink">Codoctopus</span>
          </header>
          <nav className="flex gap-1 border-b border-border px-4 py-2 sm:hidden">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    'rounded-lg px-3 py-1.5 text-sm font-medium',
                    isActive ? 'bg-accent/10 text-accent' : 'text-ink-secondary',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <main className="flex-1 px-6 py-6">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}

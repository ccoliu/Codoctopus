import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { NewRun } from './pages/NewRun'
import { RunDetail } from './pages/RunDetail'
import { RunsList } from './pages/RunsList'
import { Settings } from './pages/Settings'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<NewRun />} />
        <Route path="/runs" element={<RunsList />} />
        <Route path="/runs/:id" element={<RunDetail />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

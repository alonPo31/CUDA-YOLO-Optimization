import { useState, useEffect } from 'react'

const API = 'http://localhost:5000'

export default function App() {
  const [status, setStatus] = useState({ fps: 0, object_count: 0, nms_enabled: true })
  const [connected, setConnected] = useState(false)

  // משיכת סטטוס מה-Flask כל 500ms
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API}/status`)
        const data = await res.json()
        setStatus(data)
        setConnected(true)
      } catch {
        setConnected(false)
      }
    }, 500)
    return () => clearInterval(interval)
  }, [])

  const toggleNMS = async () => {
    try {
      const res = await fetch(`${API}/toggle_nms`, { method: 'POST' })
      const data = await res.json()
      setStatus(prev => ({ ...prev, nms_enabled: data.nms_enabled }))
    } catch {}
  }

  return (
    <div className="app">
      <header className="header">
        <h1>CUDA YOLO Detection</h1>
        <span className={`connection-badge ${connected ? 'connected' : 'disconnected'}`}>
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </header>

      <main className="main">
        <div className="stream-wrapper">
          <img
            className="stream"
            src={`${API}/video_feed`}
            alt="Live detection stream"
          />
        </div>
      </main>

      <footer className="footer">
        <div className="stat-group">
          <div className="stat">
            <span className="stat-label">FPS</span>
            <span className="stat-value">{status.fps}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Objects</span>
            <span className="stat-value">{status.object_count}</span>
          </div>
        </div>

        <button
          className={`nms-btn ${status.nms_enabled ? 'on' : 'off'}`}
          onClick={toggleNMS}
        >
          NMS: {status.nms_enabled ? 'ON' : 'OFF'}
        </button>
      </footer>
    </div>
  )
}

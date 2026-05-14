import { useState, useEffect, useRef } from 'react'

const API = 'http://localhost:5000'

export default function App() {
  const [mode, setMode] = useState('select')
  const [status, setStatus] = useState({
    fps: 0,
    object_count: 0,
    nms_enabled: true,
    tracking_enabled: true,
    conf_threshold: 0.25,
  })
  const [connected, setConnected]     = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading]     = useState(false)
  const [uploadStarted, setUploadStarted] = useState(false)
  const streamRef = useRef(null)

  const inDetection = mode === 'live' || (mode === 'upload' && uploadStarted)

  // Frame polling — only while in the detection view
  useEffect(() => {
    if (!inDetection) return
    let active = true
    let tid

    function nextFrame() {
      if (!active) return
      const tmp = new window.Image()
      tmp.onload = () => {
        if (active && streamRef.current) streamRef.current.src = tmp.src
        tid = setTimeout(nextFrame, 0)
      }
      tmp.onerror = () => {
        if (active) tid = setTimeout(nextFrame, 200)
      }
      tmp.src = `${API}/frame?t=${Date.now()}`
    }

    nextFrame()
    return () => { active = false; clearTimeout(tid) }
  }, [inDetection])

  // Status polling — only while in the detection view
  useEffect(() => {
    if (!inDetection) return
    const interval = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/status`)
        const data = await res.json()
        setStatus(data)
        setConnected(true)
      } catch {
        setConnected(false)
      }
    }, 500)
    return () => clearInterval(interval)
  }, [inDetection])

  const toggleNMS = async () => {
    try {
      const res  = await fetch(`${API}/toggle_nms`, { method: 'POST' })
      const data = await res.json()
      setStatus(prev => ({ ...prev, nms_enabled: data.nms_enabled }))
    } catch {}
  }

  const toggleTracking = async () => {
    try {
      const res  = await fetch(`${API}/toggle_tracking`, { method: 'POST' })
      const data = await res.json()
      setStatus(prev => ({ ...prev, tracking_enabled: data.tracking_enabled }))
    } catch {}
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', selectedFile)
      const res = await fetch(`${API}/upload_video`, { method: 'POST', body: fd })
      if (res.ok) setUploadStarted(true)
    } catch {}
    setUploading(false)
  }

  const handleBack = async () => {
    // If leaving upload mode, tell C++ to switch back to camera
    if (mode === 'upload' && uploadStarted) {
      try { await fetch(`${API}/switch_to_camera`, { method: 'POST' }) } catch {}
    }
    setMode('select')
    setSelectedFile(null)
    setUploadStarted(false)
  }

  const conf    = status.conf_threshold ?? 0.25
  const confPct = Math.round(conf * 100)

  // ── Select page ───────────────────────────────────────────────────────
  if (mode === 'select') {
    return (
      <div className="app">
        <header className="topbar">
          <div className="topbar-brand">
            <span className="brand-icon">◈</span>
            <span className="brand-name">CUDA YOLO Detection</span>
          </div>
        </header>

        <div className="select-page">
          <div className="select-title">Choose Detection Mode</div>
          <div className="select-subtitle">
            Full pipeline: YOLO · GPU NMS · Dynamic Threshold · Kalman Tracking
          </div>
          <div className="mode-cards">
            <div className="mode-card" onClick={() => setMode('live')}>
              <div className="mode-card-icon">📷</div>
              <div className="mode-card-name">Live Detection</div>
              <div className="mode-card-desc">
                Real-time detection from the camera using the CUDA-accelerated C++ pipeline
              </div>
            </div>
            <div className="mode-card" onClick={() => setMode('upload')}>
              <div className="mode-card-icon">📁</div>
              <div className="mode-card-name">Upload Video</div>
              <div className="mode-card-desc">
                Upload an MP4 file — the C++ backend processes it with the same CUDA pipeline
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // ── Upload page (file picker, before upload) ──────────────────────────
  if (mode === 'upload' && !uploadStarted) {
    return (
      <div className="app">
        <header className="topbar">
          <div className="topbar-brand">
            <button className="back-btn" onClick={handleBack}>← Back</button>
            <span className="brand-icon">◈</span>
            <span className="brand-name">CUDA YOLO Detection</span>
          </div>
        </header>

        <div className="upload-page">
          <div className="upload-area">
            <div className="upload-title">Upload Video File</div>
            <div className="upload-subtitle">
              Select an MP4 — the C++ CUDA backend will process it with the full pipeline
            </div>

            <label className="file-label">
              <input
                type="file"
                accept=".mp4"
                className="file-input"
                onChange={e => setSelectedFile(e.target.files[0] || null)}
              />
              <span className="file-btn">
                {selectedFile ? selectedFile.name : 'Choose MP4 File'}
              </span>
            </label>

            <button
              className="process-btn"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? 'Uploading…' : 'Process Video'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ── Detection view (live or upload) ──────────────────────────────────
  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-brand">
          <button className="back-btn" onClick={handleBack}>← Back</button>
          <span className="brand-icon">◈</span>
          <span className="brand-name">CUDA YOLO Detection</span>
        </div>
        <div className={`conn-badge ${connected ? 'conn-on' : 'conn-off'}`}>
          <span className="conn-dot" />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      <div className="layout">

        {/* Left panel — live metrics */}
        <aside className="panel panel-left">
          <div className="panel-title">Metrics</div>

          <div className="metric">
            <span className="metric-label">FPS</span>
            <span className="metric-value">{status.fps}</span>
          </div>

          <div className="divider" />

          <div className="metric">
            <span className="metric-label">Objects</span>
            <span className="metric-value">{status.object_count}</span>
          </div>

          <div className="divider" />

          <div className="metric">
            <span className="metric-label">Confidence</span>
            <span className="metric-value conf-value">{conf.toFixed(2)}</span>
            <div className="conf-bar">
              <div className="conf-fill" style={{ width: `${confPct}%` }} />
            </div>
          </div>
        </aside>

        {/* Center — video stream */}
        <main className="stream-area">
          <div className="stream-frame">
            <img
              ref={streamRef}
              className="stream"
              alt="Detection stream"
            />
          </div>
        </main>

        {/* Right panel — controls */}
        <aside className="panel panel-right">
          <div className="panel-title">Controls</div>

          <div className="control-row">
            <span className="control-label">NMS</span>
            <button
              className={`toggle ${status.nms_enabled ? 'toggle-on' : 'toggle-off'}`}
              onClick={toggleNMS}
              aria-label="Toggle NMS"
            >
              <span className="toggle-thumb" />
            </button>
          </div>

          <div className="divider" />

          <div className="control-row">
            <span className="control-label">Tracking</span>
            <button
              className={`toggle ${status.tracking_enabled ? 'toggle-on' : 'toggle-off'}`}
              onClick={toggleTracking}
              aria-label="Toggle Tracking"
            >
              <span className="toggle-thumb" />
            </button>
          </div>
        </aside>

      </div>
    </div>
  )
}

# Handoff — CUDA YOLO Optimization Session

## Goal

Get the full stack to ≥28 FPS and keep it stable during a graduation project demo.

Baseline entering this session: 25–28 FPS (up from 19–22), all pipeline threads in place, MJPEG stream sometimes going black.

---

## Current State

| Thing | Status |
|---|---|
| FPS with full stack | **28+ stable** — confirmed working |
| MJPEG stream going black | **Fixed** — JPEG validation + React watchdog |
| Corrupted-frame black flash (1 frame) | **Fixed** — JPEG magic-byte check + last_good cache |
| FPS drop when console minimized | **UNSOLVED** — workaround: use 2 screens or keep console visible |
| Connected badge shows 0 FPS as disconnected | **Fixed** |

---

## Files Changed This Session

| File | What changed |
|---|---|
| `src/main.cpp` | `SetPriorityClass`, Power Throttling opt-out, `SetWindowPos` console shrink, `NOMINMAX` fix |
| `frontend/server/app.py` | `generate_frames()` JPEG validation + last_good caching |
| `frontend/client/src/App.jsx` | `img.complete` watchdog, `connected` uses `fps > 0` |
| `run_backend.bat` | Launch helper — starts exe at RealTime priority |

---

## Changes Made This Session

### 1. MJPEG stream going black — `app.py` + `App.jsx`

**Root cause:** `cv::imwrite` on Windows is not atomic — it truncates the file, then writes the JPEG progressively. Flask could `open()` the file mid-write and read 0 bytes or a partial JPEG. The browser received a corrupt MJPEG frame, rendered black, and kept the HTTP connection open — so `onError` never fired and the existing retry logic never kicked in.

**Fix (app.py):** `generate_frames()` now validates each frame before yielding:
- Checks `len(frame_bytes) > 1000` (rejects empty/tiny reads)
- Checks `frame_bytes[:2] == b'\xff\xd8'` and `frame_bytes[-2:] == b'\xff\xd9'` (valid JPEG header + footer)
- Caches last valid frame in `last_good` — if current read fails validation, yields `last_good` instead of nothing. This prevents any black flash: the viewer sees a repeated frame for 33 ms rather than black.

**Fix (App.jsx):** Added `imgRef` ref to the `<img>` element and a 2-second watchdog `useEffect`:
```jsx
useEffect(() => {
  if (!inDetection) return
  const id = setInterval(() => {
    if (imgRef.current?.complete) setStreamKey(k => k + 1)
  }, 2000)
  return () => clearInterval(id)
}, [inDetection])
```
Chrome sets `img.complete = true` when an MJPEG connection drops silently (without firing `onError`). The watchdog catches this and forces a reconnect within 2 seconds. The existing `onError` retry handles explicit connection errors.

---

### 2. Connected badge — `App.jsx`

**Problem:** When C++ stops running, Flask keeps serving the last cached frame forever (`last_good`). The UI showed "Connected" while displaying a frozen stale frame — confusing.

**Fix:** Changed `setConnected(true)` to `setConnected(data.fps > 0)` in the status polling effect. The badge turns red ("Disconnected") as soon as FPS drops to 0, even if Flask is still up.

---

### 3. Windows priority / minimize FPS drop — `src/main.cpp`

**Problem:** FPS drops from 28+ to 19–22 when the console window is minimized.

**Root cause (confirmed):** GPU P-state. When the console window is visible on a display, the GPU's combined display+compute load keeps its boost clock active. When minimized, display load drops, the GPU's power governor reduces the clock speed, and CUDA inference slows. This is a hardware-level power management behavior.

**Attempted fixes (all ineffective for the GPU clock issue):**
- `SetPriorityClass(HIGH_PRIORITY_CLASS)` — helps CPU scheduling, not GPU clock
- `SetProcessInformation(ProcessPowerThrottling, StateMask=0)` — disables Windows 11 Efficiency Mode for CPU, but GPU clock is independent
- `start /realtime` in `run_backend.bat` — same: RealTime process priority doesn't override GPU P-states
- NVIDIA Control Panel "Prefer maximum performance" — suggested but reportedly ineffective on this hardware

**What IS in the code now (kept as belt-and-suspenders):**
```cpp
SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
// Power Throttling opt-out (Windows 11)
PROCESS_POWER_THROTTLING_STATE ppts = {};
ppts.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
ppts.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
ppts.StateMask = 0;
SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &ppts, sizeof(ppts));
// Console shrinks to 420×80 strip on startup
if (HWND h = GetConsoleWindow())
    SetWindowPos(h, HWND_BOTTOM, 0, 0, 420, 80, SWP_NOACTIVATE);
```

**Also added:** `#define NOMINMAX` before `#include <windows.h>` — required because `windows.h` defines `min`/`max` as macros that break `std::min`/`std::max` (caused C2589 build errors).

**`run_backend.bat`** at project root — launches the Release exe at RealTime priority:
```batch
start "" /realtime "src\x64\Release\YOLO_CUDA_Project.exe"
```

**Workaround for presentation:** Use two screens — console on screen 2 (visible, not minimized), browser on screen 1. FPS stays at 28+. Alternatively: the console auto-shrinks to a 420×80 strip on startup (SetWindowPos); keep it visible in a corner of screen 1 alongside the browser.

---

## Known Issues / Next Steps

| Issue | Notes |
|---|---|
| FPS drop on minimize | GPU P-state issue. Unsolved in software. Presentation workaround: 2 screens, or keep console tiny + visible. |
| `SetWindowPos` shrinks console but doesn't fill screen perfectly | Console shows as a strip at top-left. User can drag/resize manually after launch. |

---

## Architecture Summary (unchanged)

```
Camera
  └─ Capture thread: cap.read() + blobFromImage
       └─ FrameBuffer (mutex + condvar) — latest frame only, drops old
            └─ Inference thread: net.forward + decode 8400 rows + CUDA NMS
                 └─ DetectionBuffer (mutex + condvar) — latest detections only
                      └─ Tracking thread: Kalman + Hungarian + cv::rectangle/putText
                           └─ IOBuffer (mutex + condvar) — latest annotated frame only
                                └─ IO thread: cv::imwrite + status.json write
                                     └─ latest_frame.jpg  ←── Flask reads at 30 fps
                                                               └─ /video_feed MJPEG
                                                                    └─ Browser <img>
```

All hand-offs between threads are non-blocking (newer data overwrites older). Inference thread is never blocked by tracking, drawing, JPEG encoding, or disk I/O.

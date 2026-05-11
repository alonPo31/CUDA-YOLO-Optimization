# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time YOLOv8 object detection accelerated with CUDA. Goal: ≥30 FPS on live camera with GPU-parallelized NMS, a dynamic adaptive confidence threshold, and Kalman Filter + Hungarian algorithm multi-object tracking.

## Build

**C++ backend — Visual Studio 2022 (v143 toolset)**

```
msbuild src\YOLO_CUDA_Project.vcxproj /p:Configuration=Debug /p:Platform=x64
msbuild src\YOLO_CUDA_Project.vcxproj /p:Configuration=Release /p:Platform=x64
```

Executable lands in `src/x64/Debug/` or `src/x64/Release/`. You can also open `src/YOLO_CUDA_Project.sln` and press Ctrl+Shift+B.

`.cu` files are compiled by the CUDA 13.1 MSBuild extension (`CUDA 13.1.props` / `CUDA 13.1.targets`) — not nvcc directly.

**Hardcoded dependency paths (in .vcxproj):**
- CUDA Toolkit 13.1 — `$(CudaToolkitIncludeDir)` / `$(CudaToolkitLibDir)`
- OpenCV 4.14.0 — Debug: `C:\opencv_workspace\build\install\`, Release: `C:\Alon\Programs\opencv\build\`
- Model: `src/yolov8m.onnx` (gitignored, must be placed manually)

**Flask backend**

```
cd frontend/server
python app.py          # serves http://localhost:5000
```

**React frontend**

```
cd frontend/client
npm run dev            # Vite dev server
```

## Architecture

### Full pipeline per frame

```
Capture thread (CPU)                  Main thread
────────────────────────────          ──────────────────────────────────────────
cap.read(frame)                       net.setInput(blob)
blobFromImage → 640×640               net.forward() → 1×84×8400 tensor
  /255, BGR→RGB                         reshape → 8400×84
  writes to FrameBuffer                 filter by confThreshold (dynamic)
                                        sort by confidence ↓
                                        runCudaNMS (nms.cu) → keep/suppress flags
                                        collect surviving boxes → detections[]

                                      Tracking (tracker.cpp / CPU)
                                        KalmanTracker::predict() on all tracks
                                        build cost matrix: cost[i][j] = 1 − IoU
                                        hungarian() → optimal assignment
                                        KalmanTracker::update() for matched pairs
                                        spawn new tracks for unmatched detections
                                        purge tracks with timeSinceUpdate > MAX_AGE
                                        draw boxes with "ID:N classname" labels

                                      write latest_frame.jpg + status.json
```

The capture thread overlaps camera I/O and `blobFromImage` preprocessing with GPU inference — this is the pipeline parallelism. Synchronization is a `std::mutex` + `std::condition_variable` on `FrameBuffer`.

### File-based IPC (C++ ↔ Flask)

| File | Written by | Read by | Purpose |
|---|---|---|---|
| `latest_frame.jpg` | C++ (every frame) | Flask `/video_feed` (MJPEG) | live stream |
| `status.json` | C++ (every frame) | Flask `/status` | FPS, object count, NMS flag |
| `frontend_config.json` | Flask `/toggle_nms` | C++ (every 15 frames) | NMS on/off toggle |

### Key data structures

**`BoundingBox`** (`nms.h`) — `{x1, y1, x2, y2, confidence}` in original image pixel space (scaled from 640-space).

**`KalmanTracker`** (`tracker.h`) — state vector `[cx, cy, w, h, vx, vy, vw, vh]` (8×1). Constant-velocity motion model. All matrix math (including Gauss-Jordan inversion) is implemented from scratch in `tracker.cpp` — no `<cmath>`, no `<algorithm>`.

**`Matrix`** (`tracker.h`) — row-major flat `vector<float>`, `at(r, c)` accessor.

### CUDA kernels (`nms.cu`)

- **`calculateIoUMatrix`** — 2D grid `(n+15)/16 × (n+15)/16`, 16×16 blocks. Fills a symmetric n×n IoU matrix.
- **`nmsKernel`** — 1 block × n threads. Thread i suppresses itself if any j < i has IoU > threshold. **Hard limit: n ≤ 1024.**
- **`runCudaNMS`** — `extern "C"` wrapper: alloc device memory → IoU matrix kernel → NMS kernel → copy results back.

### Dynamic confidence threshold

Lives in `main.cpp`, adjusted every 5 frames (hysteresis):
- Pre-NMS: lower by 0.02 if `candidates.size() < 3`
- Post-NMS: raise by 0.02 if `keptCount > 10`; lower by 0.02 if `keptCount < 1`
- Recovery: drift up by 0.01 toward 0.25 baseline when `keptCount` is 1–10 and threshold has drifted below 0.25
- Bounds: `[0.15, 0.75]`

### Tracking parameters (`main.cpp`)

- `MAX_AGE = 5` — frames a track survives without a matched detection before being deleted
- `IOU_THRESHOLD = 0.3f` — minimum IoU to accept a Hungarian match

## What's implemented

| Component | Status |
|---|---|
| YOLOv8 ONNX inference (OpenCV DNN + CUDA) | ✅ Done |
| GPU-parallelized NMS | ✅ Done |
| Camera capture loop + capture-thread pipeline parallelism | ✅ Done |
| Dynamic adaptive confidence threshold | ✅ Done |
| Kalman Filter + Hungarian algorithm tracking | ✅ Done |
| Flask MJPEG server + React frontend | ✅ Done |
| CUDA Streams for async GPU memory transfers | ❌ Not yet |

## Known limitations

- `nmsKernel` uses 1 block × n threads — crashes if `n > 1024`. Fix: bitmask NMS kernel (each thread i writes a 64-bit suppress-mask, results are OR-reduced), enabling multi-block execution without the n×n matrix.
- No `cudaGetLastError` / `cudaDeviceSynchronize` error checking anywhere in `nms.cu`.
- `tracker.cpp` must be manually added to the Visual Studio project (right-click project → Add → Existing Item) — it is not yet in `YOLO_CUDA_Project.vcxproj`.

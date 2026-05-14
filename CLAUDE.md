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

The capture thread overlaps camera I/O and `blobFromImage` preprocessing with GPU inference. Synchronization is a `std::mutex` + `std::condition_variable` on `FrameBuffer`.

### File-based IPC (C++ ↔ Flask)

| File | Written by | Read by | Purpose |
|---|---|---|---|
| `latest_frame.jpg` | C++ (every frame) | Flask `/frame` (JPEG poll) | live stream |
| `status.json` | C++ (every frame) | Flask `/status` | FPS, object count, conf threshold, toggle states |
| `frontend_config.json` | Flask `/toggle_nms`, `/toggle_tracking` | C++ (every 15 frames) | NMS + tracking on/off |

The React frontend polls `/frame` every ~33 ms (not MJPEG) and `/status` every 500 ms. Toggles POST to Flask which writes `frontend_config.json`; C++ reads it on the next sync cycle.

### Key data structures

**`BoundingBox`** (`nms.h`) — `{x1, y1, x2, y2, confidence}` in original image pixel space (scaled from 640-space).

**`KalmanTracker`** (`tracker.h`) — state vector `[cx, cy, w, h, vx, vy, vw, vh]` (8×1). Constant-velocity motion model. All matrix math (including Gauss-Jordan inversion) is implemented from scratch in `tracker.cpp` — no external math library.

**`Matrix`** (`tracker.h`) — row-major flat `vector<float>`, `at(r, c)` accessor.

### CUDA NMS kernel (`nms.cu`)

The implementation uses a **bitmask approach** with a `cudaStream_t` for async GPU operations:

- **`nms_bitmask_kernel`** — 2D grid of `(n+63)/64 × (n+63)/64` blocks, 64 threads per block. Each block handles a 64×64 tile of box comparisons. Column boxes are loaded into `__shared__` memory once per block; each thread computes IoU against all 64 column boxes and writes a 64-bit mask. Only the upper triangle is computed (IoU is symmetric). No hard limit on n.
- **`runCudaNMS`** — `extern "C"` wrapper: creates a stream → `cudaMemsetAsync` → `cudaMemcpyAsync` (H2D) → kernel → `cudaMemcpyAsync` (D2H) → `cudaStreamSynchronize` → greedy CPU bitmask scan → `cudaStreamDestroy`.

The greedy CPU scan uses a `remv` bitmask vector: iterates boxes in confidence order, keeps any box not flagged in `remv`, then OR-merges that box's mask row into `remv` to suppress its overlapping neighbors.

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
| YOLOv8 ONNX inference (OpenCV DNN + CUDA backend) | ✅ Done |
| GPU-parallelized bitmask NMS | ✅ Done |
| CUDA Streams for async GPU memory transfers | ✅ Done |
| Camera capture thread + pipeline parallelism | ✅ Done |
| Dynamic adaptive confidence threshold | ✅ Done |
| Kalman Filter + Hungarian algorithm tracking | ✅ Done |
| Flask server + React frontend with live toggles | ✅ Done |

## Known limitations

- No `cudaGetLastError` checking in `nms.cu` — GPU errors are silent.
- `tracker.cpp` must be manually added to the Visual Studio project (right-click → Add → Existing Item) if it was removed; it is not auto-discovered by the `.vcxproj`.
- All file paths in `main.cpp` and `app.py` are hardcoded to `C:/Alon/CUDA-YOLO-Optimization/`.

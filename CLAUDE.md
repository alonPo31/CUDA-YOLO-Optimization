# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time YOLOv8 object detection accelerated with CUDA. The goal is ≥30 FPS on live video with GPU-parallelized NMS, a dynamic confidence threshold, and Kalman Filter + Hungarian algorithm multi-object tracking.

## Build

**Primary build method: Visual Studio 2022 (v143 toolset)**

Open `src/YOLO_CUDA_Project.sln` and build with Ctrl+Shift+B, or from the command line:

```
msbuild src\YOLO_CUDA_Project.vcxproj /p:Configuration=Debug /p:Platform=x64
msbuild src\YOLO_CUDA_Project.vcxproj /p:Configuration=Release /p:Platform=x64
```

The executable is placed in `src/x64/Debug/` or `src/x64/Release/`.

**Required dependencies (hardcoded paths in .vcxproj):**
- CUDA Toolkit 13.1 — via `$(CudaToolkitIncludeDir)` / `$(CudaToolkitLibDir)`
- OpenCV 4.14.0 — Debug: `C:\opencv_workspace\build\install\`, Release: `C:\Alon\Programs\opencv\build\`
- Model file: `src/yolov8m.onnx` (gitignored, must be placed manually)

`.cu` files are compiled by the CUDA 13.1 MSBuild extension (`CUDA 13.1.props` / `CUDA 13.1.targets`), not nvcc directly.

## Architecture

### Current pipeline (single image)

```
main.cpp                          nms.cu
────────────────────────────      ──────────────────────────────────
Load yolov8m.onnx via              calculateIoUMatrix kernel
  cv::dnn (CUDA backend)   ──→      (2D grid, 16×16 blocks)
                                     computes full n×n IoU matrix
Preprocess: blobFromImage          nmsKernel
  640×640, /255, BGR→RGB   ──→      (1D, 1 block × n threads)
                                     marks suppressed boxes
Forward pass → 1×84×8400
Reshape → 8400×84           ──→   runCudaNMS (extern "C" wrapper)
  [cx,cy,w,h | 80 scores]           alloc → IoU matrix → NMS → copy back

Filter by confThreshold (0.25)
Sort by confidence (descending)
Run runCudaNMS (threshold 0.45)
Draw kept boxes (nmsResult == 0)
```

### Data structures

**`BoundingBox`** (defined in `nms.h`):
```cpp
struct BoundingBox { float x1, y1, x2, y2, confidence; };
```
Coordinates are in original image pixel space (scaled from 640×640 model space). NMS result: `0` = kept, `1` = suppressed.

### CUDA kernel details

- **`calculateIoUMatrix`**: 2D grid of `(n+15)/16 × (n+15)/16` blocks, each 16×16 threads. Computes symmetric n×n IoU matrix.
- **`nmsKernel`**: 1 block × n threads. Each thread i checks all j < i; if any IoU > threshold it marks box i as suppressed. **Works only for small n** (≤1024 threads per block).

### YOLOv8 output decoding

Output tensor `1×84×8400` is reshaped to `84×8400` then transposed to `8400×84`. Each row: `[cx, cy, w, h, score_0 … score_79]`. Coordinates are in 640-space and must be scaled by `img.cols/640` and `img.rows/640`.

## What's implemented vs. planned

| Component | Status |
|---|---|
| YOLOv8 ONNX inference (OpenCV DNN + CUDA) | ✅ Done |
| GPU-parallelized NMS | ✅ Done |
| Video/camera capture loop | ❌ Not yet |
| Dynamic confidence threshold | ❌ Not yet |
| Kalman Filter + Hungarian algorithm tracking | ❌ Not yet |
| CUDA Streams pipeline parallelism | ❌ Not yet |
| React + Flask frontend | ❌ Not yet |

## Known limitations to address

- `nmsKernel` uses 1 block × n threads — will crash if `n > 1024`. Upgrade path: replace with a bitmask NMS kernel (each thread i writes a 64-bit bitmask of which boxes it suppresses, then results are OR-reduced), which supports multi-block execution and avoids the n×n matrix entirely.
- No CUDA error checking (`cudaGetLastError`) anywhere in `nms.cu`.
- Image paths in `main.cpp` are hardcoded absolute strings — needs to become video-loop input.
- Class labels are not decoded (only confidence score is displayed, not class name). The 80 COCO classes in YOLOv8 score order are: `person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush`.

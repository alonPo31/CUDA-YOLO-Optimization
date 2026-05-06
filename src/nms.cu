#include "nms.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#define THREADS_PER_BLOCK 64

// 1. Device helper: computes IoU between two boxes entirely on the GPU
__device__ inline float calculateIoU_internal(BoundingBox a, BoundingBox b) {
    float x_left   = max(a.x1, b.x1);
    float y_top    = max(a.y1, b.y1);
    float x_right  = min(a.x2, b.x2);
    float y_bottom = min(a.y2, b.y2);

    float width  = max(0.0f, x_right  - x_left);
    float height = max(0.0f, y_bottom - y_top);
    float inter  = width * height;

    float areaA     = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB     = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = areaA + areaB - inter;

    if (union_area <= 0) return 0.0f;
    return inter / union_area;
}

// 2. Bitmask NMS kernel with shared memory
//
// Grid layout: (blocks x blocks) tiles, each tile is 64x64 comparisons.
// Each block handles one tile; each thread owns one row box and compares
// it against all 64 column boxes loaded into shared memory.
// Output: mask_matrix[row * grid_width + col_block] is a 64-bit mask where
// bit i = 1 means the row box overlaps with column box i above the IoU threshold.
__global__ void nms_bitmask_kernel(const int n, const float threshold,
                                   const BoundingBox* boxes,
                                   unsigned long long* mask_matrix) {
    // Shared memory holds the 64 column boxes for the entire block to reuse.
    // Each column box is fetched from global memory once, then read 64 times —
    // placing it here avoids 64x the global memory traffic.
    __shared__ BoundingBox shared_boxes[THREADS_PER_BLOCK];

    const int col_block = blockIdx.x;
    const int row_block = blockIdx.y;

    // Symmetry optimization: IoU(A,B) == IoU(B,A), so only compute the upper
    // triangle of the comparison matrix. Blocks below the diagonal exit immediately.
    if (row_block > col_block) return;

    const int thread_id = threadIdx.x;
    const int row_start = row_block * THREADS_PER_BLOCK;
    const int col_start = col_block * THREADS_PER_BLOCK;

    // Collective load: each thread brings one column box into shared memory
    if (col_start + thread_id < n) {
        shared_boxes[thread_id] = boxes[col_start + thread_id];
    }

    // Barrier: no thread proceeds until all 64 boxes are loaded
    __syncthreads();

    if (row_start + thread_id < n) {
        const BoundingBox cur_box = boxes[row_start + thread_id]; // lives in a register
        unsigned long long mask = 0; // 64-bit result: bit i = overlaps with col box i

        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            int target_idx = col_start + i;

            // Only compare box j against box i when j > i (sorted descending by confidence)
            if (target_idx < n && (row_start + thread_id) < target_idx) {
                if (calculateIoU_internal(cur_box, shared_boxes[i]) > threshold) {
                    mask |= (1ULL << i); // set bit i
                }
            }
        }

        // Write this thread's 64-bit mask to global memory for the CPU to scan
        const int grid_width = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        mask_matrix[(row_start + thread_id) * grid_width + col_block] = mask;
    }
}

// 3. Host wrapper: allocates GPU memory, launches the kernel via a CUDA stream,
//    then runs the greedy CPU scan on the resulting bitmask.
extern "C" void runCudaNMS(BoundingBox* boxes, int n, float threshold, int* results) {
    if (n <= 0) return;

    // A CUDA stream is a dedicated command queue for the GPU.
    // Operations submitted to it (memcpy, kernels) are queued asynchronously —
    // the CPU does not block after each submission; it continues immediately.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    BoundingBox*        d_boxes;
    unsigned long long* d_mask_matrix;

    int    blocks    = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t mask_size = (size_t)n * blocks * sizeof(unsigned long long);

    // Memory allocation stays synchronous — happens once per call, cost is negligible
    cudaMalloc(&d_boxes,       n * sizeof(BoundingBox));
    cudaMalloc(&d_mask_matrix, mask_size);

    // Async zero-fill: queued on the stream, CPU moves on immediately
    cudaMemsetAsync(d_mask_matrix, 0, mask_size, stream);

    // Async H2D transfer: boxes CPU -> GPU, queued after the memset on the same stream
    cudaMemcpyAsync(d_boxes, boxes, n * sizeof(BoundingBox), cudaMemcpyHostToDevice, stream);

    // Kernel launch on the stream — queued after the H2D copy, so data is guaranteed ready
    // Third arg (0): no dynamic shared memory needed
    // Fourth arg (stream): routes execution through our stream
    dim3 grid(blocks, blocks);
    dim3 threads(THREADS_PER_BLOCK);
    nms_bitmask_kernel<<<grid, threads, 0, stream>>>(n, threshold, d_boxes, d_mask_matrix);

    // Async D2H transfer: mask results GPU -> CPU, queued after the kernel
    std::vector<unsigned long long> h_mask(n * blocks);
    cudaMemcpyAsync(h_mask.data(), d_mask_matrix, mask_size, cudaMemcpyDeviceToHost, stream);

    // Synchronization point: CPU blocks here until the entire stream has finished.
    // Everything above ran asynchronously; only now is h_mask guaranteed to hold valid data.
    cudaStreamSynchronize(stream);

    // Greedy CPU scan — O(n * blocks) bitwise OR operations
    // remv tracks which boxes have been condemned by a higher-confidence kept box
    std::vector<unsigned long long> remv(blocks, 0);

    for (int i = 0; i < n; i++) {
        int n_block  = i / THREADS_PER_BLOCK;
        int n_offset = i % THREADS_PER_BLOCK;

        if (!(remv[n_block] & (1ULL << n_offset))) {
            results[i] = 0; // keep

            // OR this box's mask row into remv — marks all boxes it overlaps as suppressed
            unsigned long long* p = &h_mask[i * blocks];
            for (int j = n_block; j < blocks; j++) {
                remv[j] |= p[j];
            }
        } else {
            results[i] = 1; // suppress
        }
    }

    cudaFree(d_boxes);
    cudaFree(d_mask_matrix);
    cudaStreamDestroy(stream); // must destroy to avoid resource leak
}

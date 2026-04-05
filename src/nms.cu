#include "nms.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 1. פונקציית עזר מתמטית (חייבת להיות ראשונה)
__device__ float calculateIoU(BoundingBox a, BoundingBox b) {
    float x_left = max(a.x1, b.x1);
    float y_top = max(a.y1, b.y1);
    float x_right = min(a.x2, b.x2);
    float y_bottom = min(a.y2, b.y2);

    float intersection_width = max(0.0f, x_right - x_left);
    float intersection_height = max(0.0f, y_bottom - y_top);
    float intersection_area = intersection_width * intersection_height;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = areaA + areaB - intersection_area;

    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

// 2. קרנל לחישוב המטריצה
__global__ void calculateIoUMatrix(BoundingBox* d_boxes, float* d_iouMatrix, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        d_iouMatrix[row * n + col] = calculateIoU(d_boxes[row], d_boxes[col]);
    }
}

// 3. קרנל ה-NMS (חייב להופיע לפני שקוראים לו ב-runCudaNMS)
__global__ void nmsKernel(float* d_iouMatrix, bool* d_isSuppressed, int n, float threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        d_isSuppressed[i] = false;
        for (int j = 0; j < i; j++) {
            if (d_iouMatrix[i * n + j] > threshold) {
                d_isSuppressed[i] = true;
                break;
            }
        }
    }
}

// 4. פונקציית השער (Wrapper) - האחרונה בקובץ
extern "C" void runCudaNMS(BoundingBox* boxes, int n, float threshold, int* results) {
    BoundingBox* d_boxes;
    float* d_iouMatrix;
    bool* d_isSuppressed;

    // הקצאת זיכרון ב-GPU
    cudaMalloc(&d_boxes, n * sizeof(BoundingBox));
    cudaMalloc(&d_iouMatrix, n * n * sizeof(float));
    cudaMalloc(&d_isSuppressed, n * sizeof(bool));

    // העתקה ל-GPU
    cudaMemcpy(d_boxes, boxes, n * sizeof(BoundingBox), cudaMemcpyHostToDevice);

    // שלב א: חישוב מטריצה
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16);
    calculateIoUMatrix << <blocksPerGrid, threadsPerBlock >> > (d_boxes, d_iouMatrix, n);

    // שלב ב: הרצת NMS
    // נשתמש בבלוק אחד כרגע כי N קטן (4)
    nmsKernel << <1, n >> > (d_iouMatrix, d_isSuppressed, n, threshold);

    // העתקת תוצאות (דרך מערך זמני של bool כי המרה ל-int ב-CPU נוחה יותר)
    bool* h_tempResults = new bool[n];
    cudaMemcpy(h_tempResults, d_isSuppressed, n * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        results[i] = h_tempResults[i] ? 1 : 0;
    }

    // ניקוי
    delete[] h_tempResults;
    cudaFree(d_boxes);
    cudaFree(d_iouMatrix);
    cudaFree(d_isSuppressed);
}
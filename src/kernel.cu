#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm> // ספרייה שמכילה את פונקציית המיון std::sort

// המבנה של תיבה אחת
struct BoundingBox {
    float x1, y1, x2, y2;
    float confidence;
};
// 3. פונקציית עזר לחישוב IoU (רצה על ה-GPU)
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

__global__ void calculateIoUMatrix(BoundingBox* d_boxes, float* d_iouMatrix, int n) {
    // חישוב השורה והעמודה של הת'רד הנוכחי
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // בדיקה שלא חרגנו מגבולות המטריצה (למקרה ששלחנו יותר ת'רדים מתיבות)
    if (row < n && col < n) {
        // כל ת'רד מחשב IoU בין התיבה בשורה i לתיבה בעמודה j
        float iou = calculateIoU(d_boxes[row], d_boxes[col]);

        // שמירה במטריצה (מיוצגת כבתור מערך 1D רציף)
        d_iouMatrix[row * n + col] = iou;
    }
}
int main() {
    const int N = 4; // נבדוק על 4 תיבות
    size_t boxes_size = N * sizeof(BoundingBox);
    size_t matrix_size = N * N * sizeof(float);

    // 1. הגדרת 4 תיבות ב-CPU
    BoundingBox h_boxes[N] = {
        {100, 100, 200, 200, 0.9}, // תיבה 0
        {110, 110, 210, 210, 0.8}, // תיבה 1 (חפיפה גבוהה עם 0)
        {500, 500, 600, 600, 0.7}, // תיבה 2 (רחוקה מאוד)
        {150, 150, 250, 250, 0.6}  // תיבה 3 (חפיפה חלקית עם 0 ו-1)
    };
    // מיון התיבות לפי Confidence מהגבוה לנמוך
    std::sort(h_boxes, h_boxes + N, [](BoundingBox a, BoundingBox b) {
        return a.confidence > b.confidence;
        });

    // עכשיו אפשר להמשיך להעתקה ל-GPU כרגיל...
    float h_matrix[N * N];

    // 2. הקצאת זיכרון ב-GPU
    BoundingBox* d_boxes;
    float* d_iouMatrix;
    cudaMalloc(&d_boxes, boxes_size);
    cudaMalloc(&d_iouMatrix, matrix_size);

    // 3. העתקה ל-GPU
    cudaMemcpy(d_boxes, h_boxes, boxes_size, cudaMemcpyHostToDevice);

    // 4. הגדרת ה-Grid (כמה ת'רדים לשלוח?)
    // נשתמש בבלוק של 16x16 ת'רדים
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // 5. הרצה!
    calculateIoUMatrix << <blocksPerGrid, threadsPerBlock >> > (d_boxes, d_iouMatrix, N);

    // 6. העתקת המטריצה חזרה ל-CPU
    cudaMemcpy(h_matrix, d_iouMatrix, matrix_size, cudaMemcpyDeviceToHost);

    // 7. הדפסת המטריצה בצורה יפה
    std::cout << "IoU Matrix (" << N << "x" << N << "):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", h_matrix[i * N + j]);
        }
        std::cout << std::endl;
    }

    // ניקוי
    cudaFree(d_boxes);
    cudaFree(d_iouMatrix);
    return 0;
}
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
__global__ void nmsKernel(float* d_iouMatrix, bool* d_isSuppressed, int n, float threshold) {
    // 1. נמצא איזה ת'רד אנחנו (איזו תיבה אנחנו בודקים)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. בדיקה שלא חרגנו ממספר התיבות
    if (i < n) {
        // 1. נניח שהתיבה שורדת
        d_isSuppressed[i] = false;

        // 2. נעבור על כל התיבות שחזקות ממנה (j < i)
        for (int j = 0; j < i; j++) {
            // 3. נבדוק את החפיפה במטריצה
            if (d_iouMatrix[i * n + j] > threshold) {
                // 4. אם חפפנו יותר מדי לתיבה חזקה יותר - נפסלנו
                d_isSuppressed[i] = true;
                break; // סיימנו עם התיבה הזאת
            }
        }
    }
}

int main() {
    // --- 1. הגדרת נתונים בסיסיים ---
    const int N = 4;
    float threshold = 0.5f;

    // הגדרת 4 תיבות (שים לב שהן לא ממוינות כאן לפי Confidence)
    BoundingBox h_boxes[N] = {
        {100, 100, 200, 200, 0.90f}, // תיבה 0 (כלב)
        {110, 110, 210, 210, 0.95f}, // תיבה 1 (כלב - חופפת ל-0, חזקה יותר!)
        {500, 500, 600, 600, 0.80f}, // תיבה 2 (חתול - רחוקה)
        {150, 150, 250, 250, 0.70f}  // תיבה 3 (כלב - חופפת ל-1, חלשה יותר)
    };

    // --- 2. מיון לפי Confidence (CPU) ---
    // חובה לבצע לפני ה-NMS כדי שהתיבה הכי חזקה תהיה באינדקס 0
    std::sort(h_boxes, h_boxes + N, [](BoundingBox a, BoundingBox b) {
        return a.confidence > b.confidence;
        });

    std::cout << "Boxes sorted by confidence. Top box: " << h_boxes[0].confidence << std::endl;

    // --- 3. הכנת זיכרון ב-GPU ---
    BoundingBox* d_boxes;
    float* d_iouMatrix;
    bool* d_isSuppressed;

    cudaMalloc(&d_boxes, N * sizeof(BoundingBox));
    cudaMalloc(&d_iouMatrix, N * N * sizeof(float));
    cudaMalloc(&d_isSuppressed, N * sizeof(bool));

    // --- 4. העברת נתונים ל-GPU ---
    cudaMemcpy(d_boxes, h_boxes, N * sizeof(BoundingBox), cudaMemcpyHostToDevice);

    // --- 5. הרצת קרנל מטריצת IoU (שלב א') ---
    // נשתמש בבלוק של 16x16 ת'רדים
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    calculateIoUMatrix << <blocksPerGrid, threadsPerBlock >> > (d_boxes, d_iouMatrix, N);

    // --- 6. הרצת קרנל NMS (שלב ב') ---
    // כל ת'רד בודק תיבה אחת מול המטריצה
    nmsKernel << <1, N >> > (d_iouMatrix, d_isSuppressed, N, threshold);

    // --- 7. הבאת תוצאות חזרה ל-CPU ---
    bool h_isSuppressed[N];
    cudaMemcpy(h_isSuppressed, d_isSuppressed, N * sizeof(bool), cudaMemcpyDeviceToHost);

    // --- 8. הדפסת תוצאות סופיות ---
    std::cout << "\n--- NMS FINAL RESULTS ---" << std::endl;
    for (int i = 0; i < N; i++) {
        std::string result = h_isSuppressed[i] ? "[REJECTED]" : "[KEEP]";
        printf("Box %d: Conf: %.2f | Status: %s\n", i, h_boxes[i].confidence, result.c_str());
    }

    // --- 9. ניקוי זיכרון ---
    cudaFree(d_boxes);
    cudaFree(d_iouMatrix);
    cudaFree(d_isSuppressed);

    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}
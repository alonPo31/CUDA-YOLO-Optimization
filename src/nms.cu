#include "nms.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#define THREADS_PER_BLOCK 64

// 1. פונקציית עזר לחישוב IoU בתוך ה-GPU
__device__ inline float calculateIoU_internal(BoundingBox a, BoundingBox b) {
    float x_left = max(a.x1, b.x1);
    float y_top = max(a.y1, b.y1);
    float x_right = min(a.x2, b.x2);
    float y_bottom = min(a.y2, b.y2);

    float width = max(0.0f, x_right - x_left);
    float height = max(0.0f, y_bottom - y_top);
    float inter = width * height;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = areaA + areaB - inter;

    if (union_area <= 0) return 0.0f;
    return inter / union_area;
}

// 2. הקרנל המייעל - Bitmask NMS עם Shared Memory
__global__ void nms_bitmask_kernel(const int n, const float threshold, const BoundingBox* boxes, unsigned long long* mask_matrix) {
    // Shared Memory - "הלוח המשותף" שבו נשים 64 תיבות מהעמודה
    __shared__ BoundingBox shared_boxes[THREADS_PER_BLOCK];

    // זיהוי המיקום של הבלוק במטריצה (איזה אריח אנחנו)
    const int col_block = blockIdx.x;
    const int row_block = blockIdx.y;

    // אופטימיזציה: מטריצת החפיפות היא סימטרית, לכן נחשב רק חצי ממנה
    if (row_block > col_block) return;

    const int thread_id = threadIdx.x;
    const int row_start = row_block * THREADS_PER_BLOCK;
    const int col_start = col_block * THREADS_PER_BLOCK;

    // טעינה קולקטיבית: כל חוט מביא תיבה אחת ל-Shared Memory
    if (col_start + thread_id < n) {
        shared_boxes[thread_id] = boxes[col_start + thread_id];
    }

    // מחסום סנכרון: מחכים שכל 64 התיבות יגיעו ללוח המשותף
    __syncthreads();

    // בדיקה עבור התיבה הספציפית של החוט הזה בשורה
    if (row_start + thread_id < n) {
        const BoundingBox cur_box = boxes[row_start + thread_id];
        unsigned long long mask = 0; // המסכה שתכיל 64 החלטות בינאריות

        // הלולאה המרכזית: השוואה מול כל התיבות שב-Shared Memory
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            int target_idx = col_start + i;

            // תנאי: אנחנו משווים רק מול תיבות שנמצאות אחרינו במערך הממוין
            if (target_idx < n && (row_start + thread_id) < target_idx) {
                if (calculateIoU_internal(cur_box, shared_boxes[i]) > threshold) {
                    // הדלקת ביט i במסכה אם יש חפיפה גבוהה
                    mask |= (1ULL << i);
                }
            }
        }

        // שמירת המסכה בזיכרון הגלובלי עבור ה-CPU
        const int grid_width = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        mask_matrix[(row_start + thread_id) * grid_width + col_block] = mask;
    }
}

// 3. פונקציית ה-Wrapper שמנהלת את הכל
extern "C" void runCudaNMS(BoundingBox* boxes, int n, float threshold, int* results) {
    if (n <= 0) return;

    BoundingBox* d_boxes;
    unsigned long long* d_mask_matrix;

    // חישוב גדלים להקצאת זיכרון
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t mask_size = (size_t)n * blocks * sizeof(unsigned long long);

    // הקצאת זיכרון ב-GPU
    cudaMalloc(&d_boxes, n * sizeof(BoundingBox));
    cudaMalloc(&d_mask_matrix, mask_size);
    cudaMemset(d_mask_matrix, 0, mask_size);

    // העתקת התיבות ל-GPU
    cudaMemcpy(d_boxes, boxes, n * sizeof(BoundingBox), cudaMemcpyHostToDevice);

    // הגדרת ה-Grid וה-Threads
    dim3 grid(blocks, blocks);
    dim3 threads(THREADS_PER_BLOCK);

    // הרצת הקרנל
    nms_bitmask_kernel << <grid, threads >> > (n, threshold, d_boxes, d_mask_matrix);

    // העתקת מטריצת הביטים חזרה ל-CPU
    std::vector<unsigned long long> h_mask(n * blocks);
    cudaMemcpy(h_mask.data(), d_mask_matrix, mask_size, cudaMemcpyDeviceToHost);

    // 4. Post-processing על ה-CPU: סריקת הביטים המהירה
    // remv הוא מערך של ביטים שמסמן אילו תיבות כבר החלטנו למחוק
    std::vector<unsigned long long> remv(blocks, 0);

    for (int i = 0; i < n; i++) {
        int n_block = i / THREADS_PER_BLOCK;
        int n_offset = i % THREADS_PER_BLOCK;

        // אם הביט של התיבה ה-i ב-remv הוא 0, סימן שהיא נשארת
        if (!(remv[n_block] & (1ULL << n_offset))) {
            results[i] = 0; // Keep

            // עכשיו נעדכן את רשימת החיסול (remv) לפי כל מה שהתיבה הזו חופפת לו
            unsigned long long* p = &h_mask[i * blocks];
            for (int j = n_block; j < blocks; j++) {
                remv[j] |= p[j]; // פעולת OR בינארית מהירה מאוד
            }
        }
        else {
            results[i] = 1; // Suppress
        }
    }

    // ניקוי זכרון
    cudaFree(d_boxes);
    cudaFree(d_mask_matrix);
}
#ifndef NMS_H
#define NMS_H

struct BoundingBox {
    float x1, y1, x2, y2;
    float confidence;
};

// הבלוק הזה מבטיח שהקומפיילר לא "יעוות" את שם הפונקציה
#ifdef __cplusplus
extern "C" {
#endif

    void runCudaNMS(BoundingBox* boxes, int n, float threshold, int* results);

#ifdef __cplusplus
}
#endif

#endif
#include <iostream>
#include <vector>
#include <algorithm>
#include "nms.h"

int main() {
    std::vector<BoundingBox> boxes = {
        {100, 100, 200, 200, 0.95f},
        {110, 110, 210, 210, 0.90f},
        {500, 500, 600, 600, 0.80f},
        {150, 150, 250, 250, 0.70f}
    };

    std::sort(boxes.begin(), boxes.end(), [](BoundingBox a, BoundingBox b) {
        return a.confidence > b.confidence;
        });

    int n = (int)boxes.size();
    std::vector<int> results(n); // נשתמש ב-int לתוצאות

    // קריאה לפונקציה עם .data() שנותן מצביע למערך הפנימי
    runCudaNMS(boxes.data(), n, 0.5f, results.data());

    std::cout << "--- NMS Results ---" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "Box " << i << ": " << (results[i] ? "[REJECTED]" : "[KEEP]") << std::endl;
    }

    return 0;
}
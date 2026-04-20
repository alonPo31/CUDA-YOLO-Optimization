#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm> // בשביל std::sort
#include "nms.h"

int main() {
    std::cout << "--- Starting Optimized YOLOv8 Inference ---" << std::endl;

    // 1. הגדרות ופרמטרים
    std::string modelPath = "C:/Alon/CUDA-YOLO-Optimization/src/yolov8m.onnx";
    float confThreshold = 0.25f; // הורדנו ל-0.25 לזיהוי רגיש יותר
    float nmsThreshold = 0.45f;

    // 2. טעינת המודל
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "ERROR: Could not load model!" << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // 3. טעינת תמונה
    cv::Mat img = cv::imread("C:/Alon/CUDA-YOLO-Optimization/src/japan.jpg");
    if (img.empty()) {
        std::cerr << "ERROR: Image not found!" << std::endl;
        return -1;
    }

    // 4. Pre-processing תקני
    // swapRB=true הופך מ-BGR ל-RGB, 1/255.0 מנרמל ל-0-1
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    // 5. הרצת המודל
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 6. פענוח פלט (YOLOv8 Output: 1 x 84 x 8400)
    cv::Mat output = outputs[0];
    if (output.dims > 2) {
        output = output.reshape(1, 84); // הופכים למטריצה של 84 שורות על 8400 עמודות
    }
    cv::transpose(output, output); // הופכים ל-8400 שורות על 84 עמודות לנוחות עבודה

    std::vector<BoundingBox> candidates;
    float scaleX = (float)img.cols / 640.0f;
    float scaleY = (float)img.rows / 640.0f;

    float* data = (float*)output.data;
    for (int i = 0; i < 8400; ++i) {
        float* row = data + (i * 84);
        float* scores = row + 4; // 4 האיברים הראשונים הם קואורדינטות, היתר ציוני מחלקות

        // מציאת המחלקה עם הציון הגבוה ביותר
        cv::Mat scoreMat(1, 80, CV_32F, scores);
        double maxConf;
        cv::Point classIdPoint;
        cv::minMaxLoc(scoreMat, 0, &maxConf, 0, &classIdPoint);

        if (maxConf > confThreshold) {
            float cx = row[0] * scaleX;
            float cy = row[1] * scaleY;
            float w = row[2] * scaleX;
            float h = row[3] * scaleY;

            BoundingBox box;
            box.x1 = cx - w / 2;
            box.y1 = cy - h / 2;
            box.x2 = cx + w / 2;
            box.y2 = cy + h / 2;
            box.confidence = (float)maxConf;
            candidates.push_back(box);
        }
    }

    if (candidates.empty()) {
        std::cout << "No objects detected." << std::endl;
        return 0;
    }

    // --- התיקון הקריטי עבור ה-CUDA NMS שלך ---
    // מיון התיבות מהביטחון הגבוה לנמוך
    std::sort(candidates.begin(), candidates.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.confidence > b.confidence;
        });

    // 7. הרצת CUDA NMS
    int n = (int)candidates.size();
    std::vector<int> nmsResults(n);
    runCudaNMS(candidates.data(), n, nmsThreshold, nmsResults.data());

    // 8. ציור תוצאות
    for (int i = 0; i < n; i++) {
        if (nmsResults[i] == 0) { // 0 אומר שהתיבה לא סוננה
            cv::Rect drawingRect(
                (int)candidates[i].x1,
                (int)candidates[i].y1,
                (int)(candidates[i].x2 - candidates[i].x1),
                (int)(candidates[i].y2 - candidates[i].y1)
            );
            cv::rectangle(img, drawingRect, cv::Scalar(0, 255, 0), 2);
            std::string label = std::to_string(candidates[i].confidence).substr(0, 4);
            cv::putText(img, label, drawingRect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }

    cv::imshow("CUDA YOLO Detection", img);
    cv::waitKey(0);

    return 0;
}
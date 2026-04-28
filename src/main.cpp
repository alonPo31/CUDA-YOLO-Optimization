#include <iostream>
#include <vector>
#include <numeric>   // בשביל std::iota
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm> // בשביל std::sort
#include <fstream>   // בשביל כתיבת קבצי סטטוס ופריימים לפרונטאנד
#include "nms.h"

// שמות 80 המחלקות של COCO לפי סדר פלט YOLOv8
static const char* CLASS_NAMES[] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
};

int main() {
    std::cout << "--- Starting Optimized YOLOv8 Real-Time Detection ---" << std::endl;

    // 1. הגדרות ופרמטרים
    const std::string modelPath      = "C:/Alon/CUDA-YOLO-Optimization/src/yolov8m.onnx";
    const std::string frameOutPath   = "C:/Alon/CUDA-YOLO-Optimization/latest_frame.jpg";
    const std::string statusOutPath  = "C:/Alon/CUDA-YOLO-Optimization/status.json";
    const std::string configInPath   = "C:/Alon/CUDA-YOLO-Optimization/frontend_config.json";
    const float confThreshold        = 0.25f; // סף ביטחון מינימלי לזיהוי
    const float nmsThreshold         = 0.45f; // סף IoU עבור NMS

    // 2. טעינת המודל
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "ERROR: Could not load model!" << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // 3. פתיחת המצלמה (0 = מצלמה ראשונה)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera!" << std::endl;
        return -1;
    }
    std::cout << "Press 'q' to quit, 'n' to toggle NMS." << std::endl;

    // יצירת החלון לפני הלולאה כדי שנוכל להוסיף trackbar
    cv::namedWindow("CUDA YOLO Detection", cv::WINDOW_AUTOSIZE);

    // טוגל NMS: 1 = פעיל, 0 = כבוי
    int nmsEnabled = 1;
    cv::createTrackbar("NMS", "CUDA YOLO Detection", &nmsEnabled, 1);

    // משתנים למדידת FPS
    double fps        = 0.0;
    int    frameCount = 0;
    double fpsTimer   = (double)cv::getTickCount();

    int configSyncCounter = 0; // סופר לסנכרון הגדרות מהפרונטאנד

    // 4. לולאת הוידאו הראשית
    while (true) {
        cv::Mat img;
        if (!cap.read(img) || img.empty()) break;

        // קריאת הגדרות מהפרונטאנד כל 15 פריימים
        if (++configSyncCounter >= 15) {
            configSyncCounter = 0;
            std::ifstream configFile(configInPath);
            if (configFile.is_open()) {
                std::string line, content;
                while (std::getline(configFile, line)) content += line;
                // פרסור פשוט: מחפשים "nms_enabled":true/false
                auto pos = content.find("\"nms_enabled\":");
                if (pos != std::string::npos) {
                    bool val = content.find("true", pos) != std::string::npos;
                    nmsEnabled = val ? 1 : 0;
                    cv::setTrackbarPos("NMS", "CUDA YOLO Detection", nmsEnabled);
                }
            }
        }

        float scaleX = (float)img.cols / 640.0f;
        float scaleY = (float)img.rows / 640.0f;

        // 5. Pre-processing תקני
        // swapRB=true הופך מ-BGR ל-RGB, 1/255.0 מנרמל ל-0-1
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);

        // 6. הרצת המודל
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 7. פענוח פלט (YOLOv8 Output: 1 x 84 x 8400)
        cv::Mat output = outputs[0];
        if (output.dims > 2) {
            output = output.reshape(1, 84); // הופכים למטריצה של 84 שורות על 8400 עמודות
        }
        cv::transpose(output, output); // הופכים ל-8400 שורות על 84 עמודות לנוחות עבודה

        std::vector<BoundingBox> candidates;
        std::vector<int>         candidateClasses; // שומרים את המחלקה לכל תיבה במקביל

        float* data = (float*)output.data;
        for (int i = 0; i < 8400; ++i) {
            float* row    = data + (i * 84);
            float* scores = row + 4; // 4 האיברים הראשונים הם קואורדינטות, היתר ציוני מחלקות

            // מציאת המחלקה עם הציון הגבוה ביותר
            cv::Mat   scoreMat(1, 80, CV_32F, scores);
            double    maxConf;
            cv::Point classIdPoint;
            cv::minMaxLoc(scoreMat, nullptr, &maxConf, nullptr, &classIdPoint);

            if (maxConf > confThreshold) {
                float cx = row[0] * scaleX;
                float cy = row[1] * scaleY;
                float w  = row[2] * scaleX;
                float h  = row[3] * scaleY;

                BoundingBox box;
                box.x1         = cx - w / 2;
                box.y1         = cy - h / 2;
                box.x2         = cx + w / 2;
                box.y2         = cy + h / 2;
                box.confidence = (float)maxConf;
                candidates.push_back(box);
                candidateClasses.push_back(classIdPoint.x); // שמירת אינדקס המחלקה
            }
        }

        int keptCount = 0; // מספר האובייקטים שנשמרו אחרי NMS (לסטטוס פרונטאנד)
        if (!candidates.empty()) {
            // --- מיון לפי ביטחון תוך שמירת סנכרון עם מערך המחלקות ---
            // במקום למיין ישירות, ממיינים מערך אינדקסים
            std::vector<int> order(candidates.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return candidates[a].confidence > candidates[b].confidence;
            });

            std::vector<BoundingBox> sorted_boxes(candidates.size());
            std::vector<int>         sorted_classes(candidates.size());
            for (int i = 0; i < (int)order.size(); i++) {
                sorted_boxes[i]   = candidates[order[i]];
                sorted_classes[i] = candidateClasses[order[i]];
            }

            // 8. הרצת CUDA NMS (אם פעיל)
            int n = (int)sorted_boxes.size();
            std::vector<int> nmsResults(n, 0); // ברירת מחדל: כל התיבות נשמרות
            if (nmsEnabled)
                runCudaNMS(sorted_boxes.data(), n, nmsThreshold, nmsResults.data());

            // 9. ציור תוצאות
            for (int i = 0; i < n; i++) {
                if (nmsResults[i] != 0) continue; // 0 אומר שהתיבה לא סוננה
                keptCount++;

                cv::Rect drawingRect(
                    (int)sorted_boxes[i].x1,
                    (int)sorted_boxes[i].y1,
                    (int)(sorted_boxes[i].x2 - sorted_boxes[i].x1),
                    (int)(sorted_boxes[i].y2 - sorted_boxes[i].y1)
                );

                cv::rectangle(img, drawingRect, cv::Scalar(0, 255, 0), 2);

                // תווית: שם המחלקה + ציון ביטחון
                std::string label = std::string(CLASS_NAMES[sorted_classes[i]]) + " "
                                  + std::to_string(sorted_boxes[i].confidence).substr(0, 4);
                cv::putText(img, label, drawingRect.tl() + cv::Point(0, -4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }

        // 10. חישוב והצגת FPS
        frameCount++;
        double elapsed = ((double)cv::getTickCount() - fpsTimer) / cv::getTickFrequency();
        if (elapsed >= 1.0) {
            fps        = frameCount / elapsed;
            frameCount = 0;
            fpsTimer   = (double)cv::getTickCount();
        }
        cv::putText(img, "FPS: " + std::to_string((int)fps),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        // הצגת סטטוס NMS על המסך
        std::string nmsLabel = nmsEnabled ? "NMS: ON" : "NMS: OFF";
        cv::Scalar  nmsColor = nmsEnabled ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(img, nmsLabel, cv::Point(10, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, nmsColor, 2);

        cv::imshow("CUDA YOLO Detection", img);

        // כתיבת פריים וסטטוס לפרונטאנד
        cv::imwrite(frameOutPath, img);
        {
            std::ofstream sf(statusOutPath);
            sf << "{\"fps\":" << (int)fps
               << ",\"object_count\":" << keptCount
               << ",\"nms_enabled\":" << (nmsEnabled ? "true" : "false") << "}";
        }

        int key = cv::waitKey(1);
        if (key == 'q') break;                              // 'q' לעצירה
        if (key == 'n') {                                   // 'n' לטוגל NMS
            nmsEnabled = 1 - nmsEnabled;
            cv::setTrackbarPos("NMS", "CUDA YOLO Detection", nmsEnabled); // סנכרון ה-trackbar
        }
    }

    // 11. ניקוי
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

#include <iostream>
#include <vector>
#include <numeric>             // std::iota
#include <thread>              // std::thread — dedicated capture thread
#include <mutex>               // std::mutex — guards shared frame buffer
#include <condition_variable>  // std::condition_variable — wakes main thread when frame is ready
#include <atomic>              // std::atomic<bool> — thread-safe stop flag
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>           // std::sort
#include <fstream>             // file writes for the web frontend
#include "nms.h"
#include "tracker.h"

// COCO class names in YOLOv8 output order
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

// Shared buffer between the capture thread and the inference (main) thread.
// The capture thread writes; the main thread reads.
struct FrameBuffer {
    cv::Mat img;           // original frame — used for drawing boxes at the end
    cv::Mat blob;          // preprocessed blob — ready input for the network
    float   scaleX = 1.f;  // scale factor X: maps from 640-space back to original resolution
    float   scaleY = 1.f;  // scale factor Y
    bool    ready  = false; // true when a new frame is waiting to be consumed
};

int main() {
    std::cout << "--- Starting Optimized YOLOv8 Real-Time Detection (CUDA Streams) ---" << std::endl;

    // 1. Parameters
    const std::string modelPath     = "C:/Alon/CUDA-YOLO-Optimization/src/yolov8m.onnx";
    const std::string frameOutPath  = "C:/Alon/CUDA-YOLO-Optimization/latest_frame.jpg";
    const std::string statusOutPath = "C:/Alon/CUDA-YOLO-Optimization/status.json";
    const std::string configInPath  = "C:/Alon/CUDA-YOLO-Optimization/frontend_config.json";
    float confThreshold             = 0.25f; // minimum confidence to consider a detection — adjusted dynamically each frame
    const float nmsThreshold        = 0.45f; // IoU threshold for NMS suppression

    // 2. Load model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "ERROR: Could not load model!" << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // 3. Open camera (0 = default camera)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera!" << std::endl;
        return -1;
    }
    std::cout << "Press 'q' to quit, 'n' to toggle NMS." << std::endl;

    // Create window before the loop so the trackbar can be attached
    cv::namedWindow("CUDA YOLO Detection", cv::WINDOW_AUTOSIZE);

    // NMS toggle: 1 = enabled, 0 = disabled
    int nmsEnabled = 1;
    cv::createTrackbar("NMS", "CUDA YOLO Detection", &nmsEnabled, 1);

    // --- Thread synchronization primitives ---

    // mutex: only one thread holds it at a time, preventing simultaneous read/write of the buffer
    std::mutex frameMtx;

    // condition_variable: lets the main thread sleep until a new frame arrives
    // instead of busy-waiting (which would waste a CPU core)
    std::condition_variable frameCv;

    FrameBuffer       buffer;
    std::atomic<bool> running{ true }; // atomic = safe to read/write from two threads without a mutex

    // 4. Capture thread — runs in parallel with the main inference loop.
    // While the GPU is busy with inference on frame N,
    // this thread reads frame N+1 from the camera and runs blobFromImage on it.
    // That overlap is the core pipeline parallelism benefit.
    std::thread captureThread([&]() {
        while (running.load()) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                running.store(false);
                frameCv.notify_one(); // wake main thread so it doesn't wait forever
                break;
            }

            // Preprocessing runs on this thread's CPU core, overlapping with GPU inference
            float sx = (float)frame.cols / 640.0f;
            float sy = (float)frame.rows / 640.0f;
            cv::Mat b = cv::dnn::blobFromImage(
                frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false
            );

            {
                std::unique_lock<std::mutex> lock(frameMtx);
                // std::move transfers ownership without copying pixel data (~0 cost)
                buffer.img    = std::move(frame);
                buffer.blob   = std::move(b);
                buffer.scaleX = sx;
                buffer.scaleY = sy;
                buffer.ready  = true;
            } // lock releases here (RAII), allowing the main thread to acquire it

            frameCv.notify_one(); // signal that a new frame is available
        }
    });

    // FPS measurement
    double fps              = 0.0;
    int    frameCount       = 0;
    double fpsTimer         = (double)cv::getTickCount();
    int    configSyncCounter = 0;
    int    threshAdjCounter  = 0; // hysteresis: only adjust confThreshold every 10 frames

    // Tracking state — persists across frames
    const int   MAX_AGE       = 5;    // frames a track survives without a matched detection
    const float IOU_THRESHOLD = 0.3f; // minimum IoU to accept a track-detection match
    std::vector<KalmanTracker> tracks;

    // 5. Main inference loop
    while (running.load()) {

        cv::Mat img, blob;
        float   scaleX, scaleY;

        // Wait for the capture thread to produce a frame
        {
            std::unique_lock<std::mutex> lock(frameMtx);

            // Releases the mutex and sleeps until: new frame ready OR stop signal.
            // Releasing the mutex while sleeping lets the capture thread keep writing.
            frameCv.wait(lock, [&] { return buffer.ready || !running.load(); });

            if (!running.load()) break;

            // Take ownership of the frame — the capture thread can immediately write the next one
            img    = std::move(buffer.img);
            blob   = std::move(buffer.blob);
            scaleX = buffer.scaleX;
            scaleY = buffer.scaleY;
            buffer.ready = false;
        } // mutex releases here — capture thread unblocked while we run inference below

        // Sync frontend config every 15 frames
        if (++configSyncCounter >= 15) {
            configSyncCounter = 0;
            std::ifstream configFile(configInPath);
            if (configFile.is_open()) {
                std::string line, content;
                while (std::getline(configFile, line)) content += line;
                auto pos = content.find("\"nms_enabled\":");
                if (pos != std::string::npos) {
                    bool val = content.find("true", pos) != std::string::npos;
                    nmsEnabled = val ? 1 : 0;
                    cv::setTrackbarPos("NMS", "CUDA YOLO Detection", nmsEnabled);
                }
            }
        }

        // 6. Run the model on the GPU.
        // The capture thread is already preparing frame N+1 on a CPU core right now.
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 7. Decode YOLOv8 output tensor: 1 x 84 x 8400
        cv::Mat output = outputs[0];
        if (output.dims > 2) {
            output = output.reshape(1, 84); // reshape to 84 rows x 8400 cols
        }
        cv::transpose(output, output); // transpose to 8400 rows x 84 cols for easy row access

        std::vector<BoundingBox> candidates;
        std::vector<int>         candidateClasses;

        float* data = (float*)output.data;
        for (int i = 0; i < 8400; ++i) {
            float* row    = data + (i * 84);
            float* scores = row + 4; // first 4 values are cx,cy,w,h; remainder are class scores

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
                candidateClasses.push_back(classIdPoint.x);
            }
        }

        int keptCount = 0; // number of boxes surviving NMS (reported to the frontend)
        std::vector<BoundingBox> detections;
        std::vector<int>         detClasses;
        if (!candidates.empty()) {
            // Sort by confidence descending using an index array to keep candidateClasses in sync
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

            // 8. Run CUDA NMS (internally uses a CUDA stream for async GPU operations)
            int n = (int)sorted_boxes.size();
            std::vector<int> nmsResults(n, 0); // default: keep all (used when NMS is off)
            if (nmsEnabled)
                runCudaNMS(sorted_boxes.data(), n, nmsThreshold, nmsResults.data());

            // 9. Collect surviving boxes for the tracking stage
            for (int i = 0; i < n; i++) {
                if (nmsResults[i] != 0) continue;
                keptCount++;
                detections.push_back(sorted_boxes[i]);
                detClasses.push_back(sorted_classes[i]);
            }
        }

        // ── 10. Tracking: Kalman prediction + Hungarian assignment ───────────────

        int nT = (int)tracks.size();
        int nD = (int)detections.size();

        // Step 1: advance every existing track's state one frame forward
        for (auto& t : tracks)
            t.predict();

        std::vector<bool> detMatched(nD, false);

        if (nT > 0 && nD > 0) {
            // Step 2: build cost matrix — cost[i][j] = 1 - IoU(predicted box i, detection j)
            std::vector<std::vector<float>> costMatrix(nT, std::vector<float>(nD, 1.0f));
            for (int i = 0; i < nT; i++) {
                BoundingBox pred = tracks[i].getBox();
                for (int j = 0; j < nD; j++)
                    costMatrix[i][j] = 1.0f - computeIoU(pred, detections[j]);
            }

            // Step 3: find the globally optimal assignment
            std::vector<int> assignment = hungarian(costMatrix, nT, nD);

            // Step 4: update matched tracks; reject weak IoU matches
            for (int i = 0; i < nT; i++) {
                int j = assignment[i];
                if (j != -1 && (1.0f - costMatrix[i][j]) >= IOU_THRESHOLD) {
                    tracks[i].update(detections[j]);
                    tracks[i].classId = detClasses[j];
                    detMatched[j] = true;
                }
            }
        }

        // Step 5: spawn a new track for every unmatched detection
        for (int j = 0; j < nD; j++) {
            if (!detMatched[j]) {
                KalmanTracker newTrack;
                newTrack.init(detections[j]);
                newTrack.classId = detClasses[j];
                tracks.push_back(newTrack);
            }
        }

        // Step 6: purge tracks that have gone too long without a match
        {
            std::vector<KalmanTracker> live;
            for (auto& t : tracks)
                if (t.timeSinceUpdate <= MAX_AGE)
                    live.push_back(t);
            tracks = live;
        }

        // Step 7: draw every track that was matched this frame (timeSinceUpdate == 0)
        for (const auto& t : tracks) {
            if (t.timeSinceUpdate > 0) continue;
            BoundingBox box = t.getBox();
            cv::Rect r(
                (int)box.x1, (int)box.y1,
                (int)(box.x2 - box.x1), (int)(box.y2 - box.y1)
            );
            cv::rectangle(img, r, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID:" + std::to_string(t.id) + " "
                              + std::string(CLASS_NAMES[t.classId]);
            cv::putText(img, label, r.tl() + cv::Point(0, -4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // Dynamic threshold adjustment with hysteresis (every 5 frames to prevent flickering)
        if (++threshAdjCounter >= 5) {
            threshAdjCounter = 0;
            // Pre-NMS: only lower threshold if barely anything is detected
            if ((int)candidates.size() < 3)
                confThreshold = std::max(confThreshold - 0.02f, 0.15f);
            // Post-NMS: adjust threshold based on kept count
            if (keptCount > 10)
                confThreshold = std::min(confThreshold + 0.02f, 0.75f);
            else if (keptCount < 1)
                confThreshold = std::max(confThreshold - 0.02f, 0.15f);
            else if (confThreshold < 0.25f)
                confThreshold = std::min(confThreshold + 0.01f, 0.25f); // recovery: drift back up to 0.25
        }

        // 10. FPS counter — updates once per second
        frameCount++;
        double elapsed = ((double)cv::getTickCount() - fpsTimer) / cv::getTickFrequency();
        if (elapsed >= 1.0) {
            fps        = frameCount / elapsed;
            frameCount = 0;
            fpsTimer   = (double)cv::getTickCount();
        }
        cv::putText(img, "FPS: " + std::to_string((int)fps),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        cv::putText(img, "CONF: " + std::to_string(confThreshold).substr(0, 4),
                    cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 200, 0), 2);

        std::string nmsLabel = nmsEnabled ? "NMS: ON" : "NMS: OFF";
        cv::Scalar  nmsColor = nmsEnabled ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(img, nmsLabel, cv::Point(10, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, nmsColor, 2);

        cv::imshow("CUDA YOLO Detection", img);

        // Write frame and status for the web frontend
        cv::imwrite(frameOutPath, img);
        {
            std::ofstream sf(statusOutPath);
            sf << "{\"fps\":" << (int)fps
               << ",\"object_count\":" << keptCount
               << ",\"nms_enabled\":" << (nmsEnabled ? "true" : "false") << "}";
        }

        int key = cv::waitKey(1);
        if (key == 'q') {
            running.store(false); // tell the capture thread to stop too
            break;
        }
        if (key == 'n') {
            nmsEnabled = 1 - nmsEnabled;
            cv::setTrackbarPos("NMS", "CUDA YOLO Detection", nmsEnabled);
        }
    }

    // 11. Cleanup
    running.store(false); // ensure capture thread exits its loop
    frameCv.notify_all(); // wake it if it's sleeping in wait()
    captureThread.join(); // wait for capture thread to finish before destroying shared state
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

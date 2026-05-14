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

// Parse a quoted string value from raw JSON: finds "key":"value" and returns value.
static std::string parseJsonString(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\":";
    auto p = json.find(searchKey);
    if (p == std::string::npos) return "";
    auto q = json.find("\"", p + searchKey.size());
    if (q == std::string::npos) return "";
    auto r = json.find("\"", q + 1);
    if (r == std::string::npos) return "";
    return json.substr(q + 1, r - q - 1);
}

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

    // 3. Read initial video source from config (default: live camera)
    std::string videoSource = "camera";
    std::string videoPath   = "";
    {
        std::ifstream cf(configInPath);
        if (cf.is_open()) {
            std::string content, line;
            while (std::getline(cf, line)) content += line;
            std::string s = parseJsonString(content, "video_source");
            std::string p = parseJsonString(content, "video_path");
            if (!s.empty()) videoSource = s;
            if (!p.empty()) videoPath   = p;
        }
    }

    cv::VideoCapture cap;
    if (videoSource == "file" && !videoPath.empty())
        cap.open(videoPath);
    else
        cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video source!" << std::endl;
        return -1;
    }
    // NMS toggle: 1 = enabled, 0 = disabled
    int nmsEnabled = 1;

    // --- Thread synchronization primitives ---

    // mutex: only one thread holds it at a time, preventing simultaneous read/write of the buffer
    std::mutex frameMtx;

    // condition_variable: lets the main thread sleep until a new frame arrives
    // instead of busy-waiting (which would waste a CPU core)
    std::condition_variable frameCv;

    FrameBuffer       buffer;
    std::atomic<bool> running   { true };
    // restarting: set to true while the capture thread is being replaced after a source switch,
    // so the main loop does not exit during the brief moment running==false.
    std::atomic<bool> restarting{ false };

    // 4. Capture thread factory — creates a thread that reads from the current cap.
    // Called at startup and again each time the video source changes.
    auto startCaptureThread = [&]() -> std::thread {
        return std::thread([&]() {
            while (running.load()) {
                cv::Mat frame;
                if (!cap.read(frame) || frame.empty()) {
                    // End-of-file or camera disconnect. Only signal program exit if this
                    // is not a deliberate source-switch restart.
                    if (!restarting.load()) {
                        running.store(false);
                        frameCv.notify_one();
                    }
                    break;
                }

                // Preprocessing overlaps with GPU inference on the main thread
                float sx = (float)frame.cols / 640.0f;
                float sy = (float)frame.rows / 640.0f;
                cv::Mat b = cv::dnn::blobFromImage(
                    frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false
                );

                {
                    std::unique_lock<std::mutex> lock(frameMtx);
                    buffer.img    = std::move(frame);
                    buffer.blob   = std::move(b);
                    buffer.scaleX = sx;
                    buffer.scaleY = sy;
                    buffer.ready  = true;
                }
                frameCv.notify_one();
            }
        });
    };

    std::thread captureThread = startCaptureThread();

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
    bool trackingEnabled = true;

    // Per-track color palette (BGR)
    static const cv::Scalar TRACK_COLORS[] = {
        {0,255,128}, {0,128,255}, {255,0,128}, {255,255,0},
        {0,0,255},   {255,0,255}, {0,255,255}, {128,0,255},
        {255,128,0}, {128,255,0}
    };
    const int N_COLORS = 10;

    // 5. Main inference loop
    while (true) {

        cv::Mat img, blob;
        float   scaleX, scaleY;

        // Wait for the capture thread to produce a frame.
        // Only exit (break) when running==false AND we are not mid-restart.
        {
            std::unique_lock<std::mutex> lock(frameMtx);
            frameCv.wait(lock, [&] {
                return buffer.ready || (!running.load() && !restarting.load());
            });

            if (!running.load() && !restarting.load()) break;

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
                }
                auto pos2 = content.find("\"tracking_enabled\":");
                if (pos2 != std::string::npos)
                    trackingEnabled = content.find("true", pos2) != std::string::npos;

                // Check if the web UI switched the video source (e.g. user uploaded an MP4)
                std::string newSource = parseJsonString(content, "video_source");
                std::string newPath   = parseJsonString(content, "video_path");
                if (newSource.empty()) newSource = "camera";

                if (newSource != videoSource || newPath != videoPath) {
                    std::cout << "Source switch: " << newSource << " " << newPath << std::endl;
                    videoSource = newSource;
                    videoPath   = newPath;

                    // Stop the old capture thread
                    restarting.store(true);
                    running.store(false);
                    frameCv.notify_all();
                    captureThread.join();

                    // Open the new source
                    cap.release();
                    if (videoSource == "file" && !videoPath.empty())
                        cap.open(videoPath);
                    else
                        cap.open(0);
                    if (!cap.isOpened()) {
                        std::cerr << "Could not open new source, falling back to camera" << std::endl;
                        cap.open(0);
                        videoSource = "camera";
                        videoPath   = "";
                    }

                    // Reset per-session state for the new source
                    tracks.clear();
                    confThreshold    = 0.25f;
                    threshAdjCounter = 0;
                    fps = 0.0; frameCount = 0;
                    fpsTimer   = (double)cv::getTickCount();
                    buffer.ready = false;

                    running.store(true);
                    restarting.store(false);
                    captureThread = startCaptureThread();
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

        if (trackingEnabled) {
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
                cv::Scalar color = TRACK_COLORS[t.id % N_COLORS];
                cv::rectangle(img, r, color, 2);
                std::string label = "ID:" + std::to_string(t.id) + " "
                                  + std::string(CLASS_NAMES[t.classId]);
                cv::putText(img, label, r.tl() + cv::Point(0, -4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        } else {
            tracks.clear();
            for (int j = 0; j < nD; j++) {
                cv::Rect r(
                    (int)detections[j].x1, (int)detections[j].y1,
                    (int)(detections[j].x2 - detections[j].x1),
                    (int)(detections[j].y2 - detections[j].y1)
                );
                cv::rectangle(img, r, cv::Scalar(0, 255, 0), 2);
                cv::putText(img, CLASS_NAMES[detClasses[j]], r.tl() + cv::Point(0, -4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
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
        // Write frame and status for the web frontend
        cv::imwrite(frameOutPath, img);
        {
            std::ofstream sf(statusOutPath);
            sf << "{\"fps\":" << (int)fps
               << ",\"object_count\":" << keptCount
               << ",\"nms_enabled\":"       << (nmsEnabled      ? "true" : "false")
               << ",\"tracking_enabled\":" << (trackingEnabled ? "true" : "false")
               << ",\"conf_threshold\":"   << confThreshold << "}";
        }
    }

    // 11. Cleanup
    running.store(false); // ensure capture thread exits its loop
    frameCv.notify_all(); // wake it if it's sleeping in wait()
    captureThread.join(); // wait for capture thread to finish before destroying shared state
    cap.release();
    return 0;
}

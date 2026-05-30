#include <iostream>
#include <vector>
#include <numeric>             // std::iota
#include <thread>              // std::thread — dedicated capture thread
#include <mutex>               // std::mutex — guards shared frame buffer
#include <condition_variable>  // std::condition_variable — wakes main thread when frame is ready
#include <atomic>              // std::atomic<bool> — thread-safe stop flag
#include <chrono>              // sleep_until / steady_clock — pace file playback to native FPS
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>           // std::sort
#include <fstream>             // file writes for the web frontend
#include "nms.h"
#include "tracker.h"
#define NOMINMAX               // prevent windows.h from defining min/max macros (breaks std::min/max)
#include <windows.h>           // SetPriorityClass — prevent OS throttling when window is minimized

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

// Parse a boolean value bounded to its own key. The earlier impl searched for
// "true" from the key's position to end-of-string, which leaked across fields:
// {"nms_enabled":false,"tracking_enabled":true} reported nms as true.
static bool parseJsonBool(const std::string& json, const std::string& key, bool defaultVal) {
    std::string searchKey = "\"" + key + "\":";
    auto p = json.find(searchKey);
    if (p == std::string::npos) return defaultVal;
    auto start = p + searchKey.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
    auto end = json.find_first_of(",}", start);
    if (end == std::string::npos) end = json.size();
    std::string val = json.substr(start, end - start);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    return val == "true";
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

// Shared buffer between the inference thread and the I/O thread.
// The inference thread writes the latest annotated frame; the I/O thread encodes+writes to disk.
// If the I/O thread is slow, frames are dropped (only the newest is kept) — acceptable for display.
struct IOBuffer {
    cv::Mat frame;
    int     fps            = 0;
    int     objectCount    = 0;
    bool    nmsEnabled     = true;
    bool    trackingEnabled= true;
    float   confThreshold  = 0.25f;
    bool    ready          = false;
};

// Shared buffer between the inference thread and the tracking thread.
// Inference thread writes raw detections; tracking thread runs Kalman+Hungarian+draw concurrently.
// Detections are split by confidence for ByteTrack-style two-stage matching:
//   high — confidence >= confThreshold; matched first; can spawn new tracks
//   low  — confidence in [LOW_DET_CONF, confThreshold); only matched to tracks
//          left unmatched after the first stage; cannot spawn new tracks
struct DetectionBuffer {
    cv::Mat                  img;
    std::vector<BoundingBox> highDets;
    std::vector<int>         highCls;
    std::vector<BoundingBox> lowDets;
    std::vector<int>         lowCls;
    int   fps            = 0;
    int   keptCount      = 0; // number of HIGH-confidence boxes (reported to UI)
    bool  nmsEnabled     = true;
    bool  trackingEnabled= true;
    float confThreshold  = 0.25f;
    bool  reset          = false; // tells tracking thread to clear its track list (source switch)
    bool  ready          = false;
};

int main() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Windows 11 "Efficiency Mode" / Power Throttling silently cuts background
    // process execution speed when windows are minimized — this is what drops
    // FPS from 28 to ~19 on minimize. Explicitly opt this process out.
    {
        PROCESS_POWER_THROTTLING_STATE ppts = {};
        ppts.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        ppts.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        ppts.StateMask   = 0; // 0 = disabled
        SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                              &ppts, sizeof(ppts));
    }

    // Shrink the console to a small strip in the top-left corner so it stays
    // visible (prevents GPU P-state drop from display load change) without
    // taking up screen space during a demo.
    if (HWND h = GetConsoleWindow())
        SetWindowPos(h, HWND_BOTTOM, 0, 0, 420, 80, SWP_NOACTIVATE);

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

    // 3. Always start on the live camera.
    // Any leftover video_source from a previous session is intentionally ignored here.
    std::string videoSource = "camera";
    std::string videoPath   = "";

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
    // sourceFailed: set by the capture thread when a file source ends/fails irrecoverably.
    // The main loop reacts by falling back to the camera, keeping the exe alive.
    std::atomic<bool> sourceFailed{ false };

    // I/O thread: writes latest_frame.jpg and status.json off the critical path.
    std::mutex              ioMtx;
    std::condition_variable ioCv;
    IOBuffer                ioBuf;
    std::atomic<bool>       ioRunning{ true };

    std::thread ioThread([&]() {
        while (ioRunning.load()) {
            cv::Mat frameToWrite;
            int     fps_val, obj_val;
            bool    nms_val, trk_val;
            float   conf_val;
            {
                std::unique_lock<std::mutex> lk(ioMtx);
                ioCv.wait(lk, [&]{ return ioBuf.ready || !ioRunning.load(); });
                if (!ioRunning.load()) break;
                frameToWrite = std::move(ioBuf.frame);
                fps_val  = ioBuf.fps;
                obj_val  = ioBuf.objectCount;
                nms_val  = ioBuf.nmsEnabled;
                trk_val  = ioBuf.trackingEnabled;
                conf_val = ioBuf.confThreshold;
                ioBuf.ready = false;
            }
            cv::imwrite(frameOutPath, frameToWrite);
            {
                std::ofstream sf(statusOutPath);
                sf << "{\"fps\":"            << fps_val
                   << ",\"object_count\":"   << obj_val
                   << ",\"nms_enabled\":"    << (nms_val ? "true" : "false")
                   << ",\"tracking_enabled\":" << (trk_val ? "true" : "false")
                   << ",\"conf_threshold\":" << conf_val << "}";
            }
        }
    });

    // 4. Capture thread factory — creates a thread that reads from the current cap.
     // `isFileSource` controls EOF behavior: file sources loop back to the start
     // (and only signal `sourceFailed` if the loop also fails); camera sources
     // signal program exit on read failure as before.
    auto startCaptureThread = [&](bool isFileSource) -> std::thread {
        return std::thread([&, isFileSource]() {
            // Read the file's native FPS so playback matches the video's real duration.
            // Without this, capture runs as fast as the decoder allows, and short or
            // weird-framerate phone videos play wildly faster or slower than real time.
            double srcFps = isFileSource ? cap.get(cv::CAP_PROP_FPS) : 0.0;
            if (srcFps <= 0.0 || srcFps > 240.0) srcFps = 30.0; // fallback for missing/bogus metadata
            auto frameInterval = std::chrono::microseconds((long long)(1e6 / srcFps));
            auto nextFrameTime = std::chrono::steady_clock::now();

            while (running.load()) {
                if (isFileSource) {
                    auto now = std::chrono::steady_clock::now();
                    // If we've fallen way behind (e.g. just looped the video), reset to "now"
                    // instead of firing a burst of catch-up frames.
                    if (nextFrameTime < now - std::chrono::milliseconds(500))
                        nextFrameTime = now;
                    std::this_thread::sleep_until(nextFrameTime);
                    nextFrameTime += frameInterval;
                }

                cv::Mat frame;
                if (!cap.read(frame) || frame.empty()) {
                    // Deliberate source-switch restart — exit quietly.
                    if (restarting.load()) break;

                    if (isFileSource) {
                        // EOF on an MP4 — loop back to the start so the demo keeps running.
                        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        if (cap.read(frame) && !frame.empty()) {
                            // Loop succeeded — fall through and process this frame normally.
                        } else {
                            // File is truly broken (bad codec, corrupted, etc.) — ask the
                            // main thread to fall back to the camera instead of killing the exe.
                            sourceFailed.store(true);
                            restarting.store(true); // keeps the main loop from breaking out
                            running.store(false);
                            frameCv.notify_one();
                            break;
                        }
                    } else {
                        // Camera disconnect — signal program exit.
                        running.store(false);
                        frameCv.notify_one();
                        break;
                    }
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

    // Initial source is always the camera (see comment at videoSource declaration above).
    std::thread captureThread = startCaptureThread(false);

    // FPS measurement
    double fps              = 0.0;
    int    frameCount       = 0;
    double fpsTimer         = (double)cv::getTickCount();
    int    configSyncCounter = 0;
    int    threshAdjCounter  = 0; // hysteresis: only adjust confThreshold every 10 frames

    // Tracking parameters (also used inside the tracking thread lambda below)
    const int   MAX_AGE       = 15;   // ~500ms grace — long enough for two-stage matching to rescue
    const int   MAX_AGE_DRAW  = 5;    // draw Kalman prediction up to ~170ms past last update
    const int   MIN_HITS      = 3;    // tracks must be confirmed by 3 detections before being displayed
    const float IOU_THRESHOLD = 0.3f;
    const float LOW_DET_CONF  = 0.10f; // floor for collecting weak detections used only in second-stage matching
    const float CLASS_MISMATCH_COST = 10.0f; // assigned in cost matrix when track and detection classes differ
    bool trackingEnabled = true;

    // All boxes drawn green regardless of mode — avoids color flicker when a
    // track ID gets reassigned and keeps the visual identical across modes.
    const cv::Scalar BOX_COLOR(0, 255, 0);

    // Detection buffer: inference thread → tracking thread
    std::mutex              detMtx;
    std::condition_variable detCv;
    DetectionBuffer         detBuf;
    std::atomic<bool>       trkRunning{ true };

    // Tracking thread: runs Kalman+Hungarian+draw concurrently with the next frame's inference.
    // This removes all tracking cost from the inference critical path.
    std::thread trackingThread([&]() {
        std::vector<KalmanTracker> localTracks;

        while (trkRunning.load()) {
            cv::Mat                  img;
            std::vector<BoundingBox> highDets, lowDets;
            std::vector<int>         highCls,  lowCls;
            int   fps_v, kept_v;
            bool  nms_v, trk_v, rst_v;
            float conf_v;

            {
                std::unique_lock<std::mutex> lk(detMtx);
                detCv.wait(lk, [&]{ return detBuf.ready || !trkRunning.load(); });
                if (!trkRunning.load()) break;
                img      = std::move(detBuf.img);
                highDets = std::move(detBuf.highDets);
                highCls  = std::move(detBuf.highCls);
                lowDets  = std::move(detBuf.lowDets);
                lowCls   = std::move(detBuf.lowCls);
                fps_v    = detBuf.fps;
                kept_v   = detBuf.keptCount;
                nms_v    = detBuf.nmsEnabled;
                trk_v    = detBuf.trackingEnabled;
                conf_v   = detBuf.confThreshold;
                rst_v    = detBuf.reset;
                detBuf.ready = false;
            }

            if (rst_v) {
                localTracks.clear();
                if (img.empty()) continue; // source-switch reset signal, no frame to render
            }

            int nT = (int)localTracks.size();
            int nH = (int)highDets.size();
            int nL = (int)lowDets.size();

            if (trk_v) {
                for (auto& t : localTracks) t.predict();

                std::vector<bool> trackMatched(nT, false);
                std::vector<bool> highMatched(nH, false);

                // Stage 1: HIGH-confidence detections matched against ALL tracks.
                // Class-mismatched pairs get a prohibitive cost so they're never chosen.
                if (nT > 0 && nH > 0) {
                    std::vector<std::vector<float>> costMatrix(nT, std::vector<float>(nH, 1.0f));
                    for (int i = 0; i < nT; i++) {
                        BoundingBox pred = localTracks[i].getBox();
                        for (int j = 0; j < nH; j++) {
                            if (localTracks[i].classId != highCls[j])
                                costMatrix[i][j] = CLASS_MISMATCH_COST;
                            else
                                costMatrix[i][j] = 1.0f - computeIoU(pred, highDets[j]);
                        }
                    }
                    std::vector<int> assignment = hungarian(costMatrix, nT, nH);
                    for (int i = 0; i < nT; i++) {
                        int j = assignment[i];
                        if (j != -1 && (1.0f - costMatrix[i][j]) >= IOU_THRESHOLD) {
                            localTracks[i].update(highDets[j]);
                            // Class voting with hysteresis: record this HIGH-conf vote.
                            // Switch the displayed classId only if the argmax class holds
                            // a strict majority (>=60%) of all votes. Prevents the label
                            // flipping back-and-forth on alternating misclassifications.
                            int* hist = localTracks[i].classHistogram;
                            hist[highCls[j]]++;
                            int total = 0, best = 0;
                            for (int k = 0; k < 80; k++) {
                                total += hist[k];
                                if (hist[k] > hist[best]) best = k;
                            }
                            if (hist[best] * 5 >= total * 3) // hist[best] / total >= 0.6
                                localTracks[i].classId = best;
                            trackMatched[i] = true;
                            highMatched[j] = true;
                        }
                    }
                }

                // Stage 2: LOW-confidence detections matched against tracks still unmatched.
                // These keep flickering tracks alive when YOLO confidence dips, but cannot
                // spawn new tracks — that role is reserved for HIGH detections.
                std::vector<int> unmatchedIdx;
                for (int i = 0; i < nT; i++)
                    if (!trackMatched[i]) unmatchedIdx.push_back(i);
                int nU = (int)unmatchedIdx.size();

                if (nU > 0 && nL > 0) {
                    std::vector<std::vector<float>> costMatrix(nU, std::vector<float>(nL, 1.0f));
                    for (int u = 0; u < nU; u++) {
                        int i = unmatchedIdx[u];
                        BoundingBox pred = localTracks[i].getBox();
                        for (int j = 0; j < nL; j++) {
                            if (localTracks[i].classId != lowCls[j])
                                costMatrix[u][j] = CLASS_MISMATCH_COST;
                            else
                                costMatrix[u][j] = 1.0f - computeIoU(pred, lowDets[j]);
                        }
                    }
                    std::vector<int> assignment = hungarian(costMatrix, nU, nL);
                    for (int u = 0; u < nU; u++) {
                        int j = assignment[u];
                        if (j != -1 && (1.0f - costMatrix[u][j]) >= IOU_THRESHOLD) {
                            int i = unmatchedIdx[u];
                            localTracks[i].update(lowDets[j]);
                            // Intentionally do NOT bump classId from a low-conf detection.
                        }
                    }
                }

                // Spawn new tracks only from unmatched HIGH detections — never from LOW,
                // so weak false positives never produce flash boxes. Additionally, skip
                // any detection that significantly overlaps an existing CONFIRMED track,
                // even of a different class — this kills the chair-also-detected-as-bottle
                // duplicate-box case where class-aware matching rejects the second class
                // but spawn would otherwise create a parallel track on the same object.
                for (int j = 0; j < nH; j++) {
                    if (highMatched[j]) continue;

                    bool overlapsExisting = false;
                    for (const auto& t : localTracks) {
                        if (t.hits < MIN_HITS) continue;
                        if (computeIoU(t.getBox(), highDets[j]) > 0.5f) {
                            overlapsExisting = true;
                            break;
                        }
                    }
                    if (overlapsExisting) continue;

                    KalmanTracker newTrack;
                    newTrack.init(highDets[j]);
                    newTrack.classId = highCls[j];
                    newTrack.classHistogram[highCls[j]] = 1;
                    localTracks.push_back(newTrack);
                }

                {
                    std::vector<KalmanTracker> live;
                    for (auto& t : localTracks)
                        if (t.timeSinceUpdate <= MAX_AGE) live.push_back(t);
                    localTracks = std::move(live);
                }

                // Display only confirmed tracks (hits >= MIN_HITS) within MAX_AGE_DRAW of
                // their last update. Kalman fills the gap during brief detector drops.
                for (const auto& t : localTracks) {
                    if (t.hits < MIN_HITS)            continue;
                    if (t.timeSinceUpdate > MAX_AGE_DRAW) continue;
                    BoundingBox box = t.getBox();
                    cv::Rect r((int)box.x1, (int)box.y1,
                               (int)(box.x2 - box.x1), (int)(box.y2 - box.y1));
                    cv::rectangle(img, r, BOX_COLOR, 2);
                    std::string label = "ID:" + std::to_string(t.id) + " " + CLASS_NAMES[t.classId];
                    cv::putText(img, label, r.tl() + cv::Point(0, -4),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1);
                }
            } else {
                // Tracking off: draw HIGH detections directly (raw YOLO output).
                localTracks.clear();
                for (int j = 0; j < nH; j++) {
                    cv::Rect r((int)highDets[j].x1, (int)highDets[j].y1,
                               (int)(highDets[j].x2 - highDets[j].x1),
                               (int)(highDets[j].y2 - highDets[j].y1));
                    cv::rectangle(img, r, BOX_COLOR, 2);
                    cv::putText(img, CLASS_NAMES[highCls[j]], r.tl() + cv::Point(0, -4),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1);
                }
            }

            {
                std::unique_lock<std::mutex> lk(ioMtx);
                ioBuf.frame           = std::move(img);
                ioBuf.fps             = fps_v;
                ioBuf.objectCount     = kept_v;
                ioBuf.nmsEnabled      = nms_v;
                ioBuf.trackingEnabled = trk_v;
                ioBuf.confThreshold   = conf_v;
                ioBuf.ready           = true;
            }
            ioCv.notify_one();
        }
    });

    // 5. Main inference loop
    while (true) {

        cv::Mat img, blob;
        float   scaleX, scaleY;

        // Wait for the capture thread to produce a frame.
        // Only exit (break) when running==false AND we are not mid-restart.
        bool handleSourceFailure = false;
        {
            std::unique_lock<std::mutex> lock(frameMtx);
            frameCv.wait(lock, [&] {
                return buffer.ready || sourceFailed.load()
                    || (!running.load() && !restarting.load());
            });

            if (sourceFailed.load()) {
                handleSourceFailure = true;
            } else {
                if (!running.load() && !restarting.load()) break;

                // Take ownership of the frame — the capture thread can immediately write the next one
                img    = std::move(buffer.img);
                blob   = std::move(buffer.blob);
                scaleX = buffer.scaleX;
                scaleY = buffer.scaleY;
                buffer.ready = false;
            }
        } // mutex releases here — capture thread unblocked while we run inference below

        // A file source died on us. Don't take the program down with it —
        // fall back to the camera so live detection keeps working.
        if (handleSourceFailure) {
            sourceFailed.store(false);
            std::cout << "File source failed — falling back to camera" << std::endl;

            captureThread.join();
            cap.release();
            cap.open(0);

            videoSource = "camera";
            videoPath   = "";
            confThreshold    = 0.25f;
            threshAdjCounter = 0;
            fps = 0.0; frameCount = 0;
            fpsTimer = (double)cv::getTickCount();
            {
                std::unique_lock<std::mutex> lk(frameMtx);
                buffer.ready = false;
            }

            running.store(true);
            restarting.store(false);
            captureThread = startCaptureThread(false);

            // Push the camera state back into the config so the next config-sync
            // doesn't try to re-open the failed file in a loop.
            {
                std::ofstream cf(configInPath);
                cf << "{\"nms_enabled\":"        << (nmsEnabled      ? "true" : "false")
                   << ",\"tracking_enabled\":"  << (trackingEnabled ? "true" : "false")
                   << ",\"video_source\":\"camera\",\"video_path\":\"\"}";
            }

            // Discard stale tracks from the dead source.
            {
                std::unique_lock<std::mutex> lk(detMtx);
                detBuf.img.release();
                detBuf.highDets.clear();
                detBuf.highCls.clear();
                detBuf.lowDets.clear();
                detBuf.lowCls.clear();
                detBuf.reset = true;
                detBuf.ready = true;
            }
            detCv.notify_one();
            continue;
        }

        // Sync frontend config every 15 frames
        if (++configSyncCounter >= 15) {
            configSyncCounter = 0;
            std::ifstream configFile(configInPath);
            if (configFile.is_open()) {
                std::string line, content;
                while (std::getline(configFile, line)) content += line;
                nmsEnabled      = parseJsonBool(content, "nms_enabled",      true) ? 1 : 0;
                trackingEnabled = parseJsonBool(content, "tracking_enabled", true);

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
                        // Push camera state back so the next sync doesn't keep retrying the bad file.
                        std::ofstream cf(configInPath);
                        cf << "{\"nms_enabled\":"        << (nmsEnabled      ? "true" : "false")
                           << ",\"tracking_enabled\":"  << (trackingEnabled ? "true" : "false")
                           << ",\"video_source\":\"camera\",\"video_path\":\"\"}";
                    }

                    // Reset per-session state for the new source
                    confThreshold    = 0.25f;
                    threshAdjCounter = 0;
                    fps = 0.0; frameCount = 0;
                    fpsTimer   = (double)cv::getTickCount();
                    buffer.ready = false;

                    running.store(true);
                    restarting.store(false);
                    captureThread = startCaptureThread(videoSource == "file" && !videoPath.empty());

                    // Tell the tracking thread to discard stale tracks for the new source
                    {
                        std::unique_lock<std::mutex> lk(detMtx);
                        detBuf.img.release();
                        detBuf.highDets.clear();
                        detBuf.highCls.clear();
                        detBuf.lowDets.clear();
                        detBuf.lowCls.clear();
                        detBuf.reset = true;
                        detBuf.ready = true;
                    }
                    detCv.notify_one();
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

            float maxConf = 0.0f;
            int   classId = 0;
            for (int k = 0; k < 80; ++k) {
                if (scores[k] > maxConf) { maxConf = scores[k]; classId = k; }
            }

            // ByteTrack-style: keep everything above a low floor. Boxes with
            // conf in [LOW_DET_CONF, confThreshold) are used in stage-2 matching
            // only — they can't spawn tracks, so they don't add false-positive boxes.
            if (maxConf > LOW_DET_CONF) {
                float cx = row[0] * scaleX;
                float cy = row[1] * scaleY;
                float w  = row[2] * scaleX;
                float h  = row[3] * scaleY;

                BoundingBox box;
                box.x1         = cx - w / 2;
                box.y1         = cy - h / 2;
                box.x2         = cx + w / 2;
                box.y2         = cy + h / 2;
                box.confidence = maxConf;
                candidates.push_back(box);
                candidateClasses.push_back(classId);
            }
        }

        int keptCount = 0; // HIGH-confidence boxes surviving NMS (reported to the frontend)
        std::vector<BoundingBox> highDets, lowDets;
        std::vector<int>         highCls,  lowCls;
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

            // 9. Split surviving boxes into HIGH/LOW piles by current dynamic threshold.
            for (int i = 0; i < n; i++) {
                if (nmsResults[i] != 0) continue;
                if (sorted_boxes[i].confidence >= confThreshold) {
                    highDets.push_back(sorted_boxes[i]);
                    highCls.push_back(sorted_classes[i]);
                    keptCount++;
                } else {
                    lowDets.push_back(sorted_boxes[i]);
                    lowCls.push_back(sorted_classes[i]);
                }
            }
        }

        // Dynamic threshold adjustment with hysteresis (every 5 frames to prevent flickering).
        // Only runs when NMS is on — keptCount is meaningless when every candidate survives.
        if (nmsEnabled && ++threshAdjCounter >= 5) {
            threshAdjCounter = 0;
            if (keptCount > 10)
                confThreshold = std::min(confThreshold + 0.02f, 0.75f);
            else if (keptCount < 1)
                confThreshold = std::max(confThreshold - 0.02f, 0.15f);
            else {
                // Normal range — drift back toward 0.25 baseline from either side
                if (confThreshold < 0.25f)
                    confThreshold = std::min(confThreshold + 0.01f, 0.25f);
                else if (confThreshold > 0.25f)
                    confThreshold = std::max(confThreshold - 0.01f, 0.25f);
            }
        }

        // FPS counter — updates once per second
        frameCount++;
        double elapsed = ((double)cv::getTickCount() - fpsTimer) / cv::getTickFrequency();
        if (elapsed >= 1.0) {
            fps        = frameCount / elapsed;
            frameCount = 0;
            fpsTimer   = (double)cv::getTickCount();
        }

        // Hand frame + detections to the tracking thread (non-blocking — drops if tracking is busy)
        {
            std::unique_lock<std::mutex> lk(detMtx);
            detBuf.img            = std::move(img);
            detBuf.highDets       = std::move(highDets);
            detBuf.highCls        = std::move(highCls);
            detBuf.lowDets        = std::move(lowDets);
            detBuf.lowCls         = std::move(lowCls);
            detBuf.fps            = (int)fps;
            detBuf.keptCount      = keptCount;
            detBuf.nmsEnabled     = nmsEnabled;
            detBuf.trackingEnabled= trackingEnabled;
            detBuf.confThreshold  = confThreshold;
            detBuf.reset          = false;
            detBuf.ready          = true;
        }
        detCv.notify_one();
    }

    // 11. Cleanup — stop threads in pipeline order (capture → tracking → IO)
    running.store(false);
    frameCv.notify_all();
    captureThread.join();
    cap.release();

    trkRunning.store(false);
    detCv.notify_all();
    trackingThread.join();

    ioRunning.store(false);
    ioCv.notify_all();
    ioThread.join();
    return 0;
}

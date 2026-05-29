#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include "nms.h"

// ─── Standalone utilities ─────────────────────────────────────────────────────

float computeIoU(const BoundingBox& a, const BoundingBox& b);

// Solves the linear assignment problem (minimum cost matching).
// cost[i][j] = cost of assigning track i to detection j (typically 1 - IoU).
// Returns assignment[i] = detection index for track i, or -1 if unmatched.
std::vector<int> hungarian(std::vector<std::vector<float>> cost, int n_tracks, int n_dets);

// ─── Matrix ───────────────────────────────────────────────────────────────────

struct Matrix {
    int rows, cols;
    std::vector<float> data; // row-major flat storage

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    float&       at(int r, int c)       { return data[r * cols + c]; }
    const float& at(int r, int c) const { return data[r * cols + c]; }
};

Matrix matZero(int rows, int cols);
Matrix matIdentity(int n);
Matrix matAdd(const Matrix& A, const Matrix& B);
Matrix matSub(const Matrix& A, const Matrix& B);
Matrix matMul(const Matrix& A, const Matrix& B);
Matrix matTranspose(const Matrix& A);
Matrix matInverse(const Matrix& A); // Gauss-Jordan elimination

// ─── KalmanTracker ────────────────────────────────────────────────────────────
// Tracks one object across frames using a constant-velocity motion model.
// State vector x (8x1): [cx, cy, w, h, vx, vy, vw, vh]
// Measurement z (4x1): [cx, cy, w, h]  (what YOLO gives us each frame)

struct KalmanTracker {
    static int nextId;

    int id;              // unique track ID, assigned at birth
    int timeSinceUpdate; // frames since last matched detection
    int age;             // total frames this track has existed
    int hits;            // total successful matches (init counts as 1) — used for min_hits gating
    int classId = 0;     // COCO class index of the detection that spawned/last updated this track

    Matrix x;   // state (8x1)
    Matrix P;   // error covariance (8x8)
    Matrix F;   // state transition (8x8)
    Matrix H;   // measurement (4x8)
    Matrix Q;   // process noise covariance (8x8)
    Matrix R;   // measurement noise covariance (4x4)
    Matrix I8;  // 8x8 identity (pre-computed for update step)

    void        init(const BoundingBox& box); // birth: initialize state from first detection
    void        predict();                    // step 1 each frame: extrapolate state forward
    void        update(const BoundingBox& box); // step 2: correct state with matched detection
    BoundingBox getBox() const;               // convert state back to BoundingBox for drawing
};

#endif

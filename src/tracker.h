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

    int id;                  // unique track ID, assigned at birth
    int timeSinceUpdate;     // frames since last matched detection
    int age;                 // total frames this track has existed
    int hits;                // total successful matches (init counts as 1) — used for min_hits gating
    int classId = 0;         // displayed class — argmax of classHistogram, not just the latest detection
    int classHistogram[80] = {0}; // running vote count per COCO class across the track's lifetime

    // State vector is now 6-dimensional: [cx, cy, w, h, vx, vy].
    // w and h have NO velocity component — they are EMA-updated from measurements
    // only, never predicted forward. This kills the "box keeps growing after the
    // person stops moving" overshoot that the 8D [cx, cy, w, h, vx, vy, vw, vh]
    // model produced when an initial size jump (e.g. arms opening) seeded a large
    // vw/vh that took several frames to dampen back to zero.
    Matrix x;   // state (6x1)
    Matrix P;   // error covariance (6x6)
    Matrix F;   // state transition (6x6)
    Matrix H;   // measurement (4x6)
    Matrix Q;   // process noise covariance (6x6)
    Matrix R;   // measurement noise covariance (4x4)
    Matrix I8;  // identity matrix used in the update step — sized 6x6 now (legacy name)

    void        init(const BoundingBox& box); // birth: initialize state from first detection
    void        predict();                    // step 1 each frame: extrapolate state forward
    void        update(const BoundingBox& box); // step 2: correct state with matched detection
    BoundingBox getBox() const;               // convert state back to BoundingBox for drawing
};

#endif

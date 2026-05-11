#include "tracker.h"

int KalmanTracker::nextId = 0;

// ─── Scalar helpers ───────────────────────────────────────────────────────────

static float fabsf_manual(float x) { return x < 0.0f ? -x : x; }

static void swapf(float& a, float& b) { float t = a; a = b; b = t; }

static float minf(float a, float b) { return a < b ? a : b; }

static float maxf(float a, float b) { return a > b ? a : b; }

// ─── Matrix operations ────────────────────────────────────────────────────────

Matrix matZero(int rows, int cols) {
    return Matrix(rows, cols);
}

Matrix matIdentity(int n) {
    Matrix m(n, n);
    for (int i = 0; i < n; i++)
        m.at(i, i) = 1.0f;
    return m;
}

Matrix matAdd(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (int i = 0; i < (int)A.data.size(); i++)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

Matrix matSub(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, A.cols);
    for (int i = 0; i < (int)A.data.size(); i++)
        C.data[i] = A.data[i] - B.data[i];
    return C;
}

Matrix matMul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < B.cols; j++)
            for (int k = 0; k < A.cols; k++)
                C.at(i, j) += A.at(i, k) * B.at(k, j);
    return C;
}

Matrix matTranspose(const Matrix& A) {
    Matrix T(A.cols, A.rows);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            T.at(j, i) = A.at(i, j);
    return T;
}

// Gauss-Jordan elimination: builds augmented matrix [A | I], reduces to [I | A⁻¹]
Matrix matInverse(const Matrix& A) {
    int n = A.rows;

    Matrix aug(n, 2 * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            aug.at(i, j) = A.at(i, j);
        aug.at(i, n + i) = 1.0f;
    }

    for (int col = 0; col < n; col++) {
        // Partial pivoting: find row with largest absolute value in this column
        int pivot = col;
        for (int row = col + 1; row < n; row++)
            if (fabsf_manual(aug.at(row, col)) > fabsf_manual(aug.at(pivot, col)))
                pivot = row;

        // Swap current row with pivot row
        for (int j = 0; j < 2 * n; j++)
            swapf(aug.at(col, j), aug.at(pivot, j));

        // Scale pivot row so the diagonal becomes 1
        float div = aug.at(col, col);
        for (int j = 0; j < 2 * n; j++)
            aug.at(col, j) /= div;

        // Eliminate all other rows in this column
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            float factor = aug.at(row, col);
            for (int j = 0; j < 2 * n; j++)
                aug.at(row, j) -= factor * aug.at(col, j);
        }
    }

    Matrix inv(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv.at(i, j) = aug.at(i, n + j);
    return inv;
}

// ─── KalmanTracker ────────────────────────────────────────────────────────────

void KalmanTracker::init(const BoundingBox& box) {
    id              = nextId++;
    timeSinceUpdate = 0;
    age             = 0;

    float cx = (box.x1 + box.x2) / 2.0f;
    float cy = (box.y1 + box.y2) / 2.0f;
    float w  =  box.x2 - box.x1;
    float h  =  box.y2 - box.y1;

    x = Matrix(8, 1);
    x.at(0, 0) = cx;
    x.at(1, 0) = cy;
    x.at(2, 0) = w;
    x.at(3, 0) = h;

    // F: constant-velocity model — position advances by one frame of velocity
    F = matIdentity(8);
    F.at(0, 4) = 1.0f;
    F.at(1, 5) = 1.0f;
    F.at(2, 6) = 1.0f;
    F.at(3, 7) = 1.0f;

    // H: YOLO measures only [cx, cy, w, h] — the first 4 elements of the state
    H = Matrix(4, 8);
    H.at(0, 0) = 1.0f;
    H.at(1, 1) = 1.0f;
    H.at(2, 2) = 1.0f;
    H.at(3, 3) = 1.0f;

    // Q: process noise — position can drift slightly, velocity drift is tiny
    Q = Matrix(8, 8);
    Q.at(0, 0) = 1.0f;
    Q.at(1, 1) = 1.0f;
    Q.at(2, 2) = 1.0f;
    Q.at(3, 3) = 1.0f;
    Q.at(4, 4) = 0.01f;
    Q.at(5, 5) = 0.01f;
    Q.at(6, 6) = 0.01f;
    Q.at(7, 7) = 0.01f;

    // R: measurement noise — size (w, h) is noisier than center position
    R = Matrix(4, 4);
    R.at(0, 0) = 1.0f;
    R.at(1, 1) = 1.0f;
    R.at(2, 2) = 10.0f;
    R.at(3, 3) = 10.0f;

    // P: high initial uncertainty on velocity, lower on position
    P = Matrix(8, 8);
    P.at(0, 0) = 10.0f;
    P.at(1, 1) = 10.0f;
    P.at(2, 2) = 10.0f;
    P.at(3, 3) = 10.0f;
    P.at(4, 4) = 10000.0f;
    P.at(5, 5) = 10000.0f;
    P.at(6, 6) = 10000.0f;
    P.at(7, 7) = 10000.0f;

    I8 = matIdentity(8);
}

void KalmanTracker::predict() {
    x = matMul(F, x);
    P = matAdd(matMul(matMul(F, P), matTranspose(F)), Q);
    age++;
    timeSinceUpdate++;
}

void KalmanTracker::update(const BoundingBox& box) {
    Matrix z(4, 1);
    z.at(0, 0) = (box.x1 + box.x2) / 2.0f;
    z.at(1, 0) = (box.y1 + box.y2) / 2.0f;
    z.at(2, 0) =  box.x2 - box.x1;
    z.at(3, 0) =  box.y2 - box.y1;

    Matrix Ht = matTranspose(H);
    Matrix y  = matSub(z, matMul(H, x));
    Matrix S  = matAdd(matMul(matMul(H, P), Ht), R);
    Matrix K  = matMul(matMul(P, Ht), matInverse(S));

    x = matAdd(x, matMul(K, y));
    P = matMul(matSub(I8, matMul(K, H)), P);

    timeSinceUpdate = 0;
}

BoundingBox KalmanTracker::getBox() const {
    float cx = x.at(0, 0);
    float cy = x.at(1, 0);
    float w  = x.at(2, 0);
    float h  = x.at(3, 0);

    BoundingBox box;
    box.x1         = cx - w / 2.0f;
    box.y1         = cy - h / 2.0f;
    box.x2         = cx + w / 2.0f;
    box.y2         = cy + h / 2.0f;
    box.confidence = 0.0f;
    return box;
}

// ─── IoU ─────────────────────────────────────────────────────────────────────

float computeIoU(const BoundingBox& a, const BoundingBox& b) {
    float x_left   = maxf(a.x1, b.x1);
    float y_top    = maxf(a.y1, b.y1);
    float x_right  = minf(a.x2, b.x2);
    float y_bottom = minf(a.y2, b.y2);

    float w     = maxf(0.0f, x_right - x_left);
    float h     = maxf(0.0f, y_bottom - y_top);
    float inter = w * h;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni   = areaA + areaB - inter;

    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

// ─── Hungarian algorithm ──────────────────────────────────────────────────────

// DFS augmenting path: tries to find a free column for 'row' by reassigning
// existing matches. Returns true if a valid augmenting path was found.
static bool augment(int row, int sz, float EPS,
                    std::vector<std::vector<float>>& cost,
                    std::vector<int>& row_assign,
                    std::vector<int>& col_assign,
                    std::vector<bool>& visited) {
    for (int j = 0; j < sz; j++) {
        if (cost[row][j] < EPS && !visited[j]) {
            visited[j] = true;
            // Column j is free, or its current owner can find another column
            if (col_assign[j] == -1 || augment(col_assign[j], sz, EPS, cost, row_assign, col_assign, visited)) {
                row_assign[row] = j;
                col_assign[j]   = row;
                return true;
            }
        }
    }
    return false;
}

std::vector<int> hungarian(std::vector<std::vector<float>> cost, int n_tracks, int n_dets) {
    if (n_tracks == 0 || n_dets == 0)
        return std::vector<int>(n_tracks, -1);

    const float EPS   = 1e-6f;
    const float LARGE = 10.0f; // larger than any real IoU-based cost (costs are in [0,1])

    int sz = n_tracks > n_dets ? n_tracks : n_dets;

    // Pad cost matrix to sz x sz with LARGE so dummy entries are never preferred
    cost.resize(sz);
    for (int i = 0; i < sz; i++)
        cost[i].resize(sz, LARGE);

    // Step 1: row reduction — subtract each row's minimum
    for (int i = 0; i < sz; i++) {
        float mn = cost[i][0];
        for (int j = 1; j < sz; j++)
            if (cost[i][j] < mn) mn = cost[i][j];
        for (int j = 0; j < sz; j++)
            cost[i][j] -= mn;
    }

    // Step 2: column reduction — subtract each column's minimum
    for (int j = 0; j < sz; j++) {
        float mn = cost[0][j];
        for (int i = 1; i < sz; i++)
            if (cost[i][j] < mn) mn = cost[i][j];
        for (int i = 0; i < sz; i++)
            cost[i][j] -= mn;
    }

    std::vector<int> row_assign(sz, -1);
    std::vector<int> col_assign(sz, -1);

    // Repeat until a complete matching is found
    for (int iter = 0; iter < sz * sz; iter++) {
        // Reset matching and find maximum matching via augmenting paths
        for (int i = 0; i < sz; i++) row_assign[i] = -1;
        for (int j = 0; j < sz; j++) col_assign[j] = -1;

        for (int i = 0; i < sz; i++) {
            std::vector<bool> visited(sz, false);
            augment(i, sz, EPS, cost, row_assign, col_assign, visited);
        }

        // Count matched rows
        int matched = 0;
        for (int i = 0; i < sz; i++)
            if (row_assign[i] != -1) matched++;

        if (matched == sz) break; // full matching found

        // König's theorem: find minimum vertex cover
        // Z_rows / Z_cols: rows and columns reachable from unmatched rows
        // via alternating (zero-edge, matched-edge) paths
        std::vector<bool> Z_rows(sz, false);
        std::vector<bool> Z_cols(sz, false);

        // Seed: all unmatched rows
        std::vector<int> queue;
        for (int i = 0; i < sz; i++)
            if (row_assign[i] == -1)
                queue.push_back(i);

        // Propagate alternating paths
        for (int qi = 0; qi < (int)queue.size(); qi++) {
            int row = queue[qi];
            Z_rows[row] = true;
            for (int j = 0; j < sz; j++) {
                if (cost[row][j] < EPS && !Z_cols[j]) {
                    Z_cols[j] = true;
                    // Follow the matched edge from this column back to a row
                    if (col_assign[j] != -1 && !Z_rows[col_assign[j]])
                        queue.push_back(col_assign[j]);
                }
            }
        }

        // Find minimum value among uncovered cells (Z_rows && !Z_cols)
        float min_val = LARGE;
        for (int i = 0; i < sz; i++)
            for (int j = 0; j < sz; j++)
                if (Z_rows[i] && !Z_cols[j] && cost[i][j] < min_val)
                    min_val = cost[i][j];

        if (min_val >= LARGE - EPS) break;

        // Adjust matrix:
        // uncovered (Z_rows && !Z_cols):      subtract min_val  → creates new zeros
        // doubly covered (!Z_rows && Z_cols): add min_val       → preserves existing zeros
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                if (Z_rows[i] && !Z_cols[j])
                    cost[i][j] -= min_val;
                else if (!Z_rows[i] && Z_cols[j])
                    cost[i][j] += min_val;
            }
        }
    }

    // Extract assignments for real tracks only (ignore dummy rows/cols)
    std::vector<int> result(n_tracks, -1);
    for (int i = 0; i < n_tracks; i++) {
        int j = row_assign[i];
        if (j != -1 && j < n_dets)
            result[i] = j;
    }
    return result;
}

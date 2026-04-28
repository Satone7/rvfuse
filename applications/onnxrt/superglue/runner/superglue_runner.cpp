// superglue_runner.cpp — SuperGlue feature matching via ONNX Runtime.
//
// TWO INPUT MODES:
//   a. Synthetic mode (default): generate random keypoints + descriptors + scores
//      with fixed seed for reproducible BBV profiling
//   b. File mode: load keypoints + descriptors + scores from file
//
// ONNX Runtime inference → matching scores matrix (N × N)
// Postprocessing: Sinkhorn optimal transport (100 iterations)
// Output: match count, top-10 matches (idx_a, idx_b, confidence)
//
// Usage:
//   Synthetic: ./superglue_runner <model.onnx> --synthetic [iterations]
//   File mode: ./superglue_runner <model.onnx> --file <data_file> [iterations]

#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <string>

// ========================================================================
// Configuration
// ========================================================================

static constexpr int N_MAX = 1024;          // Maximum keypoints per image
static constexpr int DESC_DIM = 256;         // Descriptor dimension
static constexpr int KPTS_DIM = 2;           // Keypoint coordinate dimension (x, y)
static constexpr int SINKHORN_ITERS = 100;   // Sinkhorn iterations
static constexpr int SINKHORN_BORDER_ITERS = 1;  // Border (dustbin) iterations: 1 per original paper
static constexpr float MATCH_THRESHOLD = 0.2f;   // Match confidence threshold
static constexpr float DUSTBIN_SCORE = 0.0f;     // Dustbin score (learned in training, but fixed for inference)
static constexpr float SINKHORN_LAMBDA = 100.0f; // Sinkhorn lambda (inverse temperature)

// ========================================================================
// Synthetic data generation (fixed seed for reproducibility)
// ========================================================================

struct KeypointData {
    std::vector<float> kpts;    // (N_max, 2) — keypoint coordinates
    std::vector<float> scores;   // (N_max,) — keypoint scores
    std::vector<float> desc;     // (256, N_max) — descriptors (Conv1d format: B,D,N)
};

static KeypointData generateSyntheticData(int n) {
    // Fixed seed for reproducibility across runs
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> coordDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> scoreDist(0.5f, 1.0f);
    std::normal_distribution<float> descDist(0.0f, 1.0f);

    KeypointData data;
    data.kpts.resize(n * KPTS_DIM);
    data.scores.resize(n);
    data.desc.resize(DESC_DIM * n);

    for (int i = 0; i < n; i++) {
        data.kpts[i * KPTS_DIM + 0] = coordDist(rng);
        data.kpts[i * KPTS_DIM + 1] = coordDist(rng);
        data.scores[i] = scoreDist(rng);
        for (int d = 0; d < DESC_DIM; d++) {
            data.desc[d * n + i] = descDist(rng);  // Conv1d layout: (D, N)
        }
    }

    return data;
}

static KeypointData loadFromFile(const char* filepath) {
    KeypointData data;
    std::ifstream file(filepath);
    if (!file) {
        fprintf(stderr, "Error: cannot open data file %s\n", filepath);
        exit(1);
    }

    int n;
    file >> n;
    if (n > N_MAX) {
        fprintf(stderr, "Error: file has %d keypoints, max is %d\n", n, N_MAX);
        exit(1);
    }

    data.kpts.resize(N_MAX * KPTS_DIM, 0.0f);
    data.scores.resize(N_MAX, 0.0f);
    data.desc.resize(DESC_DIM * N_MAX, 0.0f);

    for (int i = 0; i < n; i++) {
        file >> data.kpts[i * KPTS_DIM + 0] >> data.kpts[i * KPTS_DIM + 1] >> data.scores[i];
        for (int d = 0; d < DESC_DIM; d++) {
            file >> data.desc[d * N_MAX + i];
        }
    }

    // Normalize coordinates
    for (int i = 0; i < n; i++) {
        data.kpts[i * KPTS_DIM + 0] /= 640.0f;  // assume VGA width
        data.kpts[i * KPTS_DIM + 1] /= 480.0f;  // assume VGA height
    }

    fprintf(stdout, "Loaded %d keypoints from %s\n", n, filepath);
    return data;
}

// ========================================================================
// Sinkhorn optimal transport
// ========================================================================

static void sinkhorn(
    const float* scores,     // (N, N) matching scores matrix
    int na, int nb,          // Actual keypoint counts per image
    int n_max,               // Padded dimension (N_max + 1 for dustbin)
    float* logAssignment,    // Output: (N_max+1, N_max+1) log assignment matrix
    std::vector<std::pair<int,int>>& matches,  // Output: matched index pairs
    std::vector<float>& confidences)           // Output: match confidences
{
    // Augment with dustbin: expand (N,N) → (N+1, N+1)
    // Scores matrix is (N_max, N_max) but we only use (na, nb) + dustbin
    int nrows = n_max;  // actually na+1
    int ncols = n_max;  // actually nb+1

    // Build augmented scores matrix with dustbin
    int totalRows = na + 1;
    int totalCols = nb + 1;
    std::vector<float> augScores(totalRows * totalCols, DUSTBIN_SCORE);

    // Copy scores to top-left block
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            augScores[i * totalCols + j] = SINKHORN_LAMBDA * scores[i * N_MAX + j];
        }
    }

    // Initialize log assignment matrix: exp(scores)
    std::vector<float> logAlpha(totalRows * totalCols);
    for (int i = 0; i < totalRows * totalCols; i++) {
        logAlpha[i] = std::exp(augScores[i]);
    }

    // Sinkhorn iterations
    std::vector<float> rowSum(totalRows);
    std::vector<float> colSum(totalCols);

    for (int iter = 0; iter < SINKHORN_ITERS; iter++) {
        // Row normalization
        for (int i = 0; i < totalRows; i++) {
            float sum = 0.0f;
            for (int j = 0; j < totalCols; j++) {
                sum += logAlpha[i * totalCols + j];
            }
            rowSum[i] = (sum > 1e-12f) ? sum : 1e-12f;
        }
        for (int i = 0; i < totalRows; i++) {
            float inv = 1.0f / rowSum[i];
            for (int j = 0; j < totalCols; j++) {
                logAlpha[i * totalCols + j] *= inv;
            }
        }

        // Column normalization
        for (int j = 0; j < totalCols; j++) {
            float sum = 0.0f;
            for (int i = 0; i < totalRows; i++) {
                sum += logAlpha[i * totalCols + j];
            }
            colSum[j] = (sum > 1e-12f) ? sum : 1e-12f;
        }
        for (int j = 0; j < totalCols; j++) {
            float inv = 1.0f / colSum[j];
            for (int i = 0; i < totalRows; i++) {
                logAlpha[i * totalCols + j] *= inv;
            }
        }
    }

    // Copy output
    for (int i = 0; i < totalRows * totalCols; i++) {
        logAssignment[i] = logAlpha[i];
    }

    // Extract matches: mutual argmax
    // For each row i (i < na), find column j (j < nb) that is the argmax of both row i and column j
    // Also check that the match is not the dustbin row/column
    matches.clear();
    confidences.clear();

    for (int i = 0; i < na; i++) {
        // Find column with max value in row i
        float maxVal = -1.0f;
        int bestCol = -1;
        for (int j = 0; j < nb; j++) {
            float val = logAlpha[i * totalCols + j];
            if (val > maxVal) {
                maxVal = val;
                bestCol = j;
            }
        }

        if (bestCol >= 0) {
            // Check if row i is also the max in column bestCol
            float maxValCol = -1.0f;
            int bestRow = -1;
            for (int r = 0; r < na; r++) {
                float val = logAlpha[r * totalCols + bestCol];
                if (val > maxValCol) {
                    maxValCol = val;
                    bestRow = r;
                }
            }

            if (bestRow == i && maxVal > MATCH_THRESHOLD) {
                matches.push_back({i, bestCol});
                confidences.push_back(maxVal);
            }
        }
    }

    // Sort matches by confidence descending
    std::vector<int> indices(matches.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&confidences](int a, int b) { return confidences[a] > confidences[b]; });

    std::vector<std::pair<int,int>> sortedMatches(matches.size());
    std::vector<float> sortedConf(confidences.size());
    for (size_t k = 0; k < indices.size(); k++) {
        sortedMatches[k] = matches[indices[k]];
        sortedConf[k] = confidences[indices[k]];
    }
    matches = std::move(sortedMatches);
    confidences = std::move(sortedConf);
}

// ========================================================================
// Main
// ========================================================================

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  Synthetic: %s <model.onnx> --synthetic [N_keypoints] [iterations]\n", argv[0]);
        fprintf(stderr, "  File mode: %s <model.onnx> --file <data_file> [iterations]\n", argv[0]);
        return 1;
    }

    const char* modelPath = argv[1];
    const char* mode = argv[2];

    bool useSynthetic = false;
    bool useFile = false;
    const char* dataFile = nullptr;
    int nKeypoints = N_MAX;
    int iterations = 10;

    if (strcmp(mode, "--synthetic") == 0) {
        useSynthetic = true;
        if (argc >= 4 && argv[3][0] != '-' && atoi(argv[3]) > 0) {
            nKeypoints = atoi(argv[3]);
            if (argc >= 5) iterations = atoi(argv[4]);
        } else if (argc >= 4) {
            iterations = atoi(argv[3]);
        }
    } else if (strcmp(mode, "--file") == 0) {
        useFile = true;
        if (argc < 4) {
            fprintf(stderr, "Error: --file requires a data file path\n");
            return 1;
        }
        dataFile = argv[3];
        if (argc >= 5) iterations = atoi(argv[4]);
    } else {
        fprintf(stderr, "Error: unknown mode '%s'. Use --synthetic or --file\n", mode);
        return 1;
    }

    if (nKeypoints > N_MAX) {
        fprintf(stderr, "Error: N=%d exceeds maximum N_MAX=%d\n", nKeypoints, N_MAX);
        return 1;
    }
    if (iterations < 1) iterations = 1;

    fprintf(stdout, "SuperGlue Runner\n");
    fprintf(stdout, "Mode: %s\n", useSynthetic ? "synthetic" : "file");
    fprintf(stdout, "N_max: %d\n", N_MAX);
    if (useSynthetic) fprintf(stdout, "Active keypoints: %d\n", nKeypoints);
    fprintf(stdout, "Model: %s\n", modelPath);
    fprintf(stdout, "Iterations: %d (1 warm-up + %d measured)\n",
            iterations, iterations - 1);

    // Generate or load data
    KeypointData data0, data1;
    if (useSynthetic) {
        // Use different seeds for images A and B (but fixed for reproducibility)
        data0 = generateSyntheticData(nKeypoints);
        // Image B: use a different seed offset for variety but reproducibility
        {
            std::mt19937 rng(12345);
            std::uniform_real_distribution<float> coordDist(0.3f, 0.7f);
            std::uniform_real_distribution<float> scoreDist(0.5f, 1.0f);
            std::normal_distribution<float> descDist(0.0f, 1.0f);

            data1.kpts.resize(nKeypoints * KPTS_DIM);
            data1.scores.resize(nKeypoints);
            data1.desc.resize(DESC_DIM * nKeypoints);

            for (int i = 0; i < nKeypoints; i++) {
                data1.kpts[i * KPTS_DIM + 0] = coordDist(rng);
                data1.kpts[i * KPTS_DIM + 1] = coordDist(rng);
                data1.scores[i] = scoreDist(rng);
                for (int d = 0; d < DESC_DIM; d++) {
                    data1.desc[d * nKeypoints + i] = descDist(rng);
                }
            }
        }
        fprintf(stdout, "Generated synthetic keypoints for image A and B (N=%d)\n", nKeypoints);
    } else {
        data0 = loadFromFile(dataFile);
        data1 = data0;  // For now, use same data for both images
    }

    // Prepare input tensors (pad to N_MAX)
    // kpts0: (1, N_MAX, 2)
    std::vector<float> kpts0(N_MAX * KPTS_DIM, 0.0f);
    std::vector<float> scores0(N_MAX, 0.0f);
    std::vector<float> desc0(DESC_DIM * N_MAX, 0.0f);
    std::vector<float> kpts1(N_MAX * KPTS_DIM, 0.0f);
    std::vector<float> scores1(N_MAX, 0.0f);
    std::vector<float> desc1(DESC_DIM * N_MAX, 0.0f);

    // Copy data (pad with zeros for unused entries)
    for (int i = 0; i < nKeypoints; i++) {
        kpts0[i * KPTS_DIM + 0] = data0.kpts[i * KPTS_DIM + 0];
        kpts0[i * KPTS_DIM + 1] = data0.kpts[i * KPTS_DIM + 1];
        scores0[i] = data0.scores[i];
        for (int d = 0; d < DESC_DIM; d++) {
            desc0[d * N_MAX + i] = data0.desc[d * nKeypoints + i];
        }

        kpts1[i * KPTS_DIM + 0] = data1.kpts[i * KPTS_DIM + 0];
        kpts1[i * KPTS_DIM + 1] = data1.kpts[i * KPTS_DIM + 1];
        scores1[i] = data1.scores[i];
        for (int d = 0; d < DESC_DIM; d++) {
            desc1[d * N_MAX + i] = data1.desc[d * nKeypoints + i];
        }
    }

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "superglue_runner");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    fprintf(stdout, "Loading model...\n");
    Ort::Session session(env, modelPath, sessionOpts);

    // Get input/output names
    size_t numInputs = session.GetInputCount();
    size_t numOutputs = session.GetOutputCount();
    fprintf(stdout, "Model has %zu inputs, %zu outputs\n", numInputs, numOutputs);

    std::vector<std::string> inputNames(numInputs);
    std::vector<std::string> outputNames(numOutputs);
    std::vector<const char*> inputNamePtrs(numInputs);
    std::vector<const char*> outputNamePtrs(numOutputs);

    for (size_t i = 0; i < numInputs; i++) {
        auto name = session.GetInputNameAllocated(i, allocator);
        inputNames[i] = name.get();
        inputNamePtrs[i] = inputNames[i].c_str();
        fprintf(stdout, "  Input %zu: %s\n", i, inputNames[i].c_str());
    }

    for (size_t i = 0; i < numOutputs; i++) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        outputNames[i] = name.get();
        outputNamePtrs[i] = outputNames[i].c_str();
        fprintf(stdout, "  Output %zu: %s\n", i, outputNames[i].c_str());
    }

    // Use static shapes to avoid stack corruption by ORT Session
    // (ORT Session uses large stack frames that can corrupt local variables under QEMU)
    static int64_t kptsShape[] = {1, N_MAX, KPTS_DIM};
    static int64_t scoresShape[] = {1, N_MAX};
    static int64_t descShape[] = {1, DESC_DIM, N_MAX};
    static const size_t kptsShapeLen = 3;
    static const size_t scoresShapeLen = 2;
    static const size_t descShapeLen = 3;

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Run inference
    fprintf(stdout, "Running inference...\n");
    std::vector<Ort::Value> lastOutputs;

    for (int iter = 0; iter < iterations; iter++) {
        fprintf(stdout, "  [%d/%d]%s\n", iter + 1, iterations,
                iter == 0 ? " (warm-up)" : "");

        // Create fresh tensors each iteration (ORT takes ownership)
        std::vector<Ort::Value> inputTensors;
        inputTensors.reserve(6);

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo, kpts0.data(), kpts0.size(), kptsShape, kptsShapeLen));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo, scores0.data(), scores0.size(), scoresShape, scoresShapeLen));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo, desc0.data(), desc0.size(), descShape, descShapeLen));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo, kpts1.data(), kpts1.size(), kptsShape, kptsShapeLen));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo, scores1.data(), scores1.size(), scoresShape, scoresShapeLen));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memInfo, desc1.data(), desc1.size(), descShape, descShapeLen));

        auto outputs = session.Run(
            Ort::RunOptions{},
            inputNamePtrs.data(), inputTensors.data(), inputTensors.size(),
            outputNamePtrs.data(), outputNamePtrs.size());

        if (iter == iterations - 1) {
            lastOutputs.clear();
            for (auto& o : outputs) {
                lastOutputs.push_back(std::move(o));
            }
        }
    }

    // Process output: scores_matrix shape (1, N, N)
    auto outputInfo = lastOutputs[0].GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();

    fprintf(stdout, "\nScores matrix shape: [");
    for (size_t d = 0; d < outputShape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(outputShape[d]));
    }
    fprintf(stdout, "]\n");

    float* scoresData = lastOutputs[0].GetTensorMutableData<float>();

    // Print scores statistics
    int scoresN = static_cast<int>(outputShape[1]);
    float scoresMin = scoresData[0], scoresMax = scoresData[0];
    double scoresSum = 0.0;
    for (int i = 0; i < scoresN * scoresN; i++) {
        scoresMin = std::min(scoresMin, scoresData[i]);
        scoresMax = std::max(scoresMax, scoresData[i]);
        scoresSum += scoresData[i];
    }
    float scoresMean = static_cast<float>(scoresSum / (scoresN * scoresN));
    fprintf(stdout, "Scores: min=%.4f max=%.4f mean=%.4f\n",
        scoresMin, scoresMax, scoresMean);

    // Sinkhorn optimal transport
    int totalRows = nKeypoints + 1;
    int totalCols = nKeypoints + 1;
    std::vector<float> logAssignment(totalRows * totalCols);
    std::vector<std::pair<int,int>> matches;
    std::vector<float> confidences;

    sinkhorn(scoresData, nKeypoints, nKeypoints, N_MAX,
             logAssignment.data(), matches, confidences);

    fprintf(stdout, "\nSinkhorn matches: %zu\n", matches.size());

    // Print top-10 matches
    int showCount = std::min(static_cast<int>(matches.size()), 10);
    fprintf(stdout, "Top %d matches:\n", showCount);
    for (int i = 0; i < showCount; i++) {
        fprintf(stdout, "  [%d] idx_a=%d idx_b=%d confidence=%.4f\n",
            i, matches[i].first, matches[i].second, confidences[i]);
    }

    // Validation
    fprintf(stdout, "\n--- Validation ---\n");
    fprintf(stdout, "Match count: %zu (expected > 0: %s)\n",
        matches.size(), matches.size() > 0 ? "PASS" : "FAIL");

    if (matches.size() > 0) {
        bool confValid = true;
        for (float c : confidences) {
            if (c < 0.0f || c > 1.0f) { confValid = false; break; }
        }
        fprintf(stdout, "Confidence in [0,1]: %s\n", confValid ? "PASS" : "FAIL");
    }

    fprintf(stdout, "Done.\n");
    return 0;
}

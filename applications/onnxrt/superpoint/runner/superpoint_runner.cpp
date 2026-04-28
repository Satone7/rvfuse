// superpoint_runner.cpp — SuperPoint interest point detection + descriptor
// extraction via ONNX Runtime.
//
// Input: grayscale image file path
// Preprocessing: convert to grayscale (if color), resize to 480x640, normalize [0,1]
// ONNX Runtime inference: two outputs (semi: heatmap, desc: descriptors)
// Postprocessing:
//   a. Softmax on semi (along channel dim, 65 classes)
//   b. Reshape to 2D heatmap (H/8 x W/8)
//   c. NMS on heatmap: local maxima within 4-pixel radius
//   d. Extract keypoint coordinates + confidence scores
//   e. Sample descriptors at keypoint locations from desc output
// Output: print keypoint count, top-10 keypoints (x,y,score), descriptor stats
//
// Usage: ./superpoint_runner <model.onnx> <image.jpg> [iterations]

#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Model input dimensions (VGA for reproducible BBV profiling)
static constexpr int INPUT_HEIGHT = 480;
static constexpr int INPUT_WIDTH = 640;
static constexpr int CELL = 8;           // Spatial output downsampling factor
static constexpr int NMS_RADIUS = 4;     // NMS suppression radius in heatmap cells
static constexpr float CONF_THRESH = 0.015f; // Confidence threshold
static constexpr int BORDER_REMOVE = 4;  // Remove keypoints this close to border

// ========================================================================
// Image preprocessing
// ========================================================================

static std::vector<unsigned char> resizeNearest(
    const unsigned char* src, int srcW, int srcH, int channels,
    int dstW, int dstH)
{
    std::vector<unsigned char> dst(dstW * dstH * channels);
    float xRatio = static_cast<float>(srcW) / dstW;
    float yRatio = static_cast<float>(srcH) / dstH;

    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int sx = std::min(static_cast<int>(x * xRatio), srcW - 1);
            int sy = std::min(static_cast<int>(y * yRatio), srcH - 1);
            int si = (sy * srcW + sx) * channels;
            int di = (y * dstW + x) * channels;
            for (int c = 0; c < channels; c++) {
                dst[di + c] = src[si + c];
            }
        }
    }
    return dst;
}

// Convert to grayscale if color, resize, normalize to [0,1], layout CHW
static void preprocess(
    const char* imagePath,
    std::vector<float>& tensorData,
    std::vector<int64_t>& /*inputShape*/)
{
    int imgW, imgH, imgC;
    unsigned char* img = stbi_load(imagePath, &imgW, &imgH, &imgC, 0);
    if (!img) {
        fprintf(stderr, "Error: cannot load image %s\n", imagePath);
        exit(1);
    }

    // Convert to grayscale if needed
    std::vector<unsigned char> gray;
    int grayW = imgW, grayH = imgH;

    if (imgC == 3) {
        gray.resize(imgW * imgH);
        for (int i = 0; i < imgW * imgH; i++) {
            // Standard luminance weights
            gray[i] = static_cast<unsigned char>(
                0.299f * img[i * 3 + 0] +
                0.587f * img[i * 3 + 1] +
                0.114f * img[i * 3 + 2]);
        }
        stbi_image_free(img);
    } else if (imgC == 1) {
        gray.assign(img, img + imgW * imgH);
        stbi_image_free(img);
    } else if (imgC == 4) {
        gray.resize(imgW * imgH);
        for (int i = 0; i < imgW * imgH; i++) {
            gray[i] = static_cast<unsigned char>(
                0.299f * img[i * 4 + 0] +
                0.587f * img[i * 4 + 1] +
                0.114f * img[i * 4 + 2]);
        }
        stbi_image_free(img);
    } else {
        stbi_image_free(img);
        fprintf(stderr, "Error: unsupported channel count %d\n", imgC);
        exit(1);
    }

    // Resize to target dimensions
    std::vector<unsigned char> resized =
        resizeNearest(gray.data(), grayW, grayH, 1, INPUT_WIDTH, INPUT_HEIGHT);

    // HWC (1ch) -> CHW, normalize to [0, 1]
    int pixels = INPUT_HEIGHT * INPUT_WIDTH;
    tensorData.resize(pixels);
    for (int i = 0; i < pixels; i++) {
        tensorData[i] = resized[i] / 255.0f;
    }

    inputShape = {1, 1, INPUT_HEIGHT, INPUT_WIDTH};
}

// ========================================================================
// Postprocessing: Softmax, NMS, descriptor sampling
// ========================================================================

// Apply softmax along channel dimension (axis=1) for semi output
// Input shape: (B, 65, Hc, Wc) where Hc=H/8, Wc=W/8
// Output: same shape with softmax applied per spatial position
static void softmaxChannel(
    const float* input, float* output,
    int batch, int channels, int Hc, int Wc)
{
    int spatial = Hc * Wc;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < spatial; s++) {
            const float* in_ptr = input + b * channels * spatial + s;
            float* out_ptr = output + b * channels * spatial + s;

            // Find max for numerical stability
            float maxVal = in_ptr[0];
            for (int c = 1; c < channels; c++) {
                maxVal = std::max(maxVal, in_ptr[c * spatial]);
            }

            // Compute exp(x - max) and sum
            float sum = 0.0f;
            for (int c = 0; c < channels; c++) {
                float v = std::exp(in_ptr[c * spatial] - maxVal);
                out_ptr[c * spatial] = v;
                sum += v;
            }

            // Normalize
            for (int c = 0; c < channels; c++) {
                out_ptr[c * spatial] /= sum;
            }
        }
    }
}

// Keypoint structure
struct Keypoint {
    float x, y;       // Image coordinates
    float score;       // Confidence after softmax + NMS
    int responseX, responseY; // Heatmap cell coordinates
};

// Extract keypoints from heatmap using NMS
// heatmap: (Hc, Wc) — single channel (dustbin removed)
// Returns vector of keypoints sorted by confidence (descending)
static std::vector<Keypoint> extractKeypoints(
    const float* heatmap, int Hc, int Wc,
    int nmsRadius, float confThresh, int borderRemove)
{
    std::vector<Keypoint> kpts;

    // Threshold and collect candidates
    for (int y = borderRemove; y < Hc - borderRemove; y++) {
        for (int x = borderRemove; x < Wc - borderRemove; x++) {
            float val = heatmap[y * Wc + x];
            if (val < confThresh) continue;

            // Check if local maximum within nmsRadius
            bool isMax = true;
            for (int dy = -nmsRadius; dy <= nmsRadius && isMax; dy++) {
                for (int dx = -nmsRadius; dx <= nmsRadius && isMax; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < Hc && nx >= 0 && nx < Wc) {
                        if (heatmap[ny * Wc + nx] > val) {
                            isMax = false;
                        }
                    }
                }
            }

            if (isMax) {
                // Convert from cell coordinates to image coordinates
                // SuperPoint uses cell-center: image_coord = cell_coord * CELL + CELL/2
                Keypoint kpt;
                kpt.x = static_cast<float>(x * CELL + CELL / 2);
                kpt.y = static_cast<float>(y * CELL + CELL / 2);
                kpt.score = val;
                kpt.responseX = x;
                kpt.responseY = y;
                kpts.push_back(kpt);
            }
        }
    }

    // Sort by confidence descending
    std::sort(kpts.begin(), kpts.end(),
        [](const Keypoint& a, const Keypoint& b) { return a.score > b.score; });

    return kpts;
}

// Sample descriptors at keypoint locations using bilinear interpolation
// desc: (B, 256, Hc, Wc) descriptor output from model
// kpts: keypoints with image coordinates
// Returns: vector of (256-dim descriptor) per keypoint
static std::vector<std::vector<float>> sampleDescriptors(
    const float* desc, int Hc, int Wc,
    const std::vector<Keypoint>& kpts)
{
    int descDim = 256;
    std::vector<std::vector<float>> descriptors;
    descriptors.reserve(kpts.size());

    for (const auto& kpt : kpts) {
        // Convert image coordinate to heatmap coordinate
        float hx = kpt.x / static_cast<float>(CELL) - 0.5f;
        float hy = kpt.y / static_cast<float>(CELL) - 0.5f;

        // Bilinear interpolation
        int x0 = std::max(0, std::min(static_cast<int>(std::floor(hx)), Wc - 1));
        int y0 = std::max(0, std::min(static_cast<int>(std::floor(hy)), Hc - 1));
        int x1 = std::min(x0 + 1, Wc - 1);
        int y1 = std::min(y0 + 1, Hc - 1);

        float wx = hx - static_cast<float>(x0);
        float wy = hy - static_cast<float>(y0);
        wx = std::max(0.0f, std::min(1.0f, wx));
        wy = std::max(0.0f, std::min(1.0f, wy));

        std::vector<float> d(descDim);
        for (int c = 0; c < descDim; c++) {
            float v00 = desc[c * Hc * Wc + y0 * Wc + x0];
            float v01 = desc[c * Hc * Wc + y0 * Wc + x1];
            float v10 = desc[c * Hc * Wc + y1 * Wc + x0];
            float v11 = desc[c * Hc * Wc + y1 * Wc + x1];
            d[c] = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                   wy * ((1 - wx) * v10 + wx * v11);
        }

        // L2 normalize the descriptor
        float norm = 0.0f;
        for (float v : d) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (float& v : d) v /= norm;
        }

        descriptors.push_back(std::move(d));
    }

    return descriptors;
}

// ========================================================================
// Main
// ========================================================================

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.onnx> <image.jpg> [iterations]\n", argv[0]);
        return 1;
    }

    const char* modelPath = argv[1];
    const char* imagePath = argv[2];
    int iterations = 10;
    if (argc >= 4) {
        iterations = atoi(argv[3]);
        if (iterations < 1) iterations = 1;
    }

    fprintf(stdout, "SuperPoint Runner\n");
    fprintf(stdout, "Target input: %dx%d (HxW)\n", INPUT_HEIGHT, INPUT_WIDTH);
    fprintf(stdout, "Model: %s\n", modelPath);
    fprintf(stdout, "Image: %s\n", imagePath);
    fprintf(stdout, "Iterations: %d (1 warm-up + %d measured)\n",
            iterations, iterations - 1);

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "superpoint_runner");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    fprintf(stdout, "Loading model...\n");
    Ort::Session session(env, modelPath, sessionOpts);

    // Get input/output names
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto semiName = session.GetOutputNameAllocated(0, allocator);
    auto descName = session.GetOutputNameAllocated(1, allocator);
    std::vector<const char*> outputNamePtrs = {semiName.get(), descName.get()};

    fprintf(stdout, "Input:  %s\n", inputName.get());
    fprintf(stdout, "Output: %s, %s\n", semiName.get(), descName.get());

    // Preprocess image
    fprintf(stdout, "Preprocessing image...\n");
    std::vector<float> tensorData;
    std::vector<int64_t> inputShapeVec;
    preprocess(imagePath, tensorData, inputShapeVec);

    // Use static array to avoid stack corruption by ORT Session constructor
    // (ORT Session uses large stack frames that can corrupt local variables under QEMU)
    static int64_t inputShape[] = {1, 1, INPUT_HEIGHT, INPUT_WIDTH};
    static const size_t inputShapeLen = 4;

    fprintf(stdout, "Input shape: [1, 1, %d, %d], %zu elements\n",
        INPUT_HEIGHT, INPUT_WIDTH, tensorData.size());

    // Create input tensor
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(),
        inputShape, inputShapeLen);

    // Run inference
    fprintf(stdout, "Running inference...\n");
    const char* inputNames[] = {inputName.get()};

    std::vector<Ort::Value> lastOutputs;

    for (int i = 0; i < iterations; i++) {
        fprintf(stdout, "  [%d/%d]%s\n", i + 1, iterations,
                i == 0 ? " (warm-up)" : "");
        auto outputs = session.Run(
            Ort::RunOptions{},
            inputNames, &inputTensor, 1,
            outputNamePtrs.data(), outputNamePtrs.size());

        if (i == iterations - 1) {
            lastOutputs.clear();
            for (auto& o : outputs) {
                lastOutputs.push_back(std::move(o));
            }
        }
    }

    // Process outputs
    // semi: (1, 65, Hc, Wc) — raw logits, need softmax
    // desc: (1, 256, Hc, Wc) — L2 normalized descriptors
    auto semiInfo = lastOutputs[0].GetTensorTypeAndShapeInfo();
    auto descInfo = lastOutputs[1].GetTensorTypeAndShapeInfo();
    auto semiShape = semiInfo.GetShape();
    auto descShape = descInfo.GetShape();

    fprintf(stdout, "\nSemi shape: [");
    for (size_t d = 0; d < semiShape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(semiShape[d]));
    }
    fprintf(stdout, "]\n");

    fprintf(stdout, "Desc shape: [");
    for (size_t d = 0; d < descShape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(descShape[d]));
    }
    fprintf(stdout, "]\n");

    float* semiData = lastOutputs[0].GetTensorMutableData<float>();
    float* descData = lastOutputs[1].GetTensorMutableData<float>();

    int batch = static_cast<int>(semiShape[0]);
    int channels = static_cast<int>(semiShape[1]);  // 65
    int Hc = static_cast<int>(semiShape[2]);         // H/8 = 60
    int Wc = static_cast<int>(semiShape[3]);         // W/8 = 80

    // Step a: Apply softmax along channel dimension
    std::vector<float> semiSoftmax(batch * channels * Hc * Wc);
    softmaxChannel(semiData, semiSoftmax.data(), batch, channels, Hc, Wc);

    // Step b: Remove dustbin channel (index 64), reshape to 2D heatmap
    // The 65 channels correspond to 8x8 grid cells + dustbin
    // After softmax, dustbin is channel 64
    // Heatmap at (y, x) = sum over 64 spatial channels reshaped to (y*8+dy, x*8+dx)
    // Simplified: use channel 0 as "pointed" score, or use the 64 non-dustbin channels
    // For keypoint extraction, we use the spatial maximum per pixel position
    std::vector<float> heatmap(Hc * Wc, 0.0f);
    for (int y = 0; y < Hc; y++) {
        for (int x = 0; x < Wc; x++) {
            // Find max over the 64 spatial channels (exclude dustbin at index 64)
            float maxVal = 0.0f;
            for (int c = 0; c < 64; c++) {
                float val = semiSoftmax[c * Hc * Wc + y * Wc + x];
                maxVal = std::max(maxVal, val);
            }
            heatmap[y * Wc + x] = maxVal;
        }
    }

    // Step c: NMS
    auto keypoints = extractKeypoints(
        heatmap.data(), Hc, Wc, NMS_RADIUS, CONF_THRESH, BORDER_REMOVE);

    fprintf(stdout, "\nDetected %zu keypoints\n", keypoints.size());

    // Step d: Print top-10 keypoints
    int showCount = std::min(static_cast<int>(keypoints.size()), 10);
    fprintf(stdout, "Top %d keypoints:\n", showCount);
    for (int i = 0; i < showCount; i++) {
        fprintf(stdout, "  [%d] x=%.1f y=%.1f score=%.4f\n",
            i, keypoints[i].x, keypoints[i].y, keypoints[i].score);
    }

    // Step e: Sample descriptors
    auto descriptors = sampleDescriptors(descData, Hc, Wc, keypoints);

    // Descriptor statistics
    if (!descriptors.empty()) {
        float normSum = 0.0f;
        float normMin = 1e9f;
        float normMax = 0.0f;
        for (const auto& d : descriptors) {
            float norm = 0.0f;
            for (float v : d) norm += v * v;
            norm = std::sqrt(norm);
            normSum += norm;
            normMin = std::min(normMin, norm);
            normMax = std::max(normMax, norm);
        }
        float avgNorm = normSum / descriptors.size();
        fprintf(stdout, "\nDescriptor statistics:\n");
        fprintf(stdout, "  Count: %zu\n", descriptors.size());
        fprintf(stdout, "  Dim: 256\n");
        fprintf(stdout, "  L2 norm: mean=%.6f min=%.6f max=%.6f\n",
            avgNorm, normMin, normMax);
    }

    // Validation summary
    fprintf(stdout, "\n--- Validation ---\n");
    fprintf(stdout, "Keypoint count: %zu (expected > 0: %s)\n",
        keypoints.size(), keypoints.size() > 0 ? "PASS" : "FAIL");

    if (!descriptors.empty()) {
        float norm = 0.0f;
        for (float v : descriptors[0]) norm += v * v;
        norm = std::sqrt(norm);
        fprintf(stdout, "First descriptor L2 norm: %.6f (expected ~1.0: %s)\n",
            norm, std::abs(norm - 1.0f) < 0.01f ? "PASS" : "FAIL");
    }

    fprintf(stdout, "Done.\n");
    return 0;
}

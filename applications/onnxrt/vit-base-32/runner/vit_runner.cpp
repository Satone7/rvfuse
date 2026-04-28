// vit_runner.cpp — ViT-Base/32 ImageNet classification via ONNX Runtime
//
// Input: RGB image file path
// Preprocessing: resize to 384x384, HWC->CHW, normalize (mean=0.5, std=0.5)
// ONNX Runtime inference: single output (logits: [1, 1000])
// Postprocessing: softmax on logits, print top-5 class indices + confidence
//
// Model: google/vit-base-patch32-384 (HuggingFace Transformers)
// Architecture: Patch embedding (Conv2d 3->768, k=32, s=32) +
//   12x Transformer Encoder (768-dim, 12 heads, d_k=64, SeqLen=145) +
//   Classification head (768->1000)
//
// Key shape differences from ViT-Base/16:
//   - Input: 384x384 (vs 224x224)
//   - Patches: 12x12=144 + CLS = 145 tokens (vs 14x14=196+1=197)
//   - 145 % VL16 = 1 tail element (vs 197 % VL16 = 5 tail elements)
//   - This gives better VL=16 utilization for attention operators
//
// Usage: ./vit_inference <model.onnx> <image.jpg> [iterations]

#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Model input dimensions
static constexpr int INPUT_HEIGHT = 384;
static constexpr int INPUT_WIDTH = 384;
static constexpr int INPUT_CHANNELS = 3;

// Normalization parameters (ViT standard)
static constexpr float MEAN[] = {0.5f, 0.5f, 0.5f};
static constexpr float STD[]  = {0.5f, 0.5f, 0.5f};

// Number of ImageNet classes
static constexpr int NUM_CLASSES = 1000;

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

// Load image, resize to 384x384, convert HWC->CHW, normalize with mean/std
static void preprocess(
    const char* imagePath,
    std::vector<float>& tensorData,
    std::vector<int64_t>& /*inputShape*/)
{
    int imgW, imgH, imgC;
    // Force 3-channel RGB output
    unsigned char* img = stbi_load(imagePath, &imgW, &imgH, &imgC, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load image %s\n", imagePath);
        exit(1);
    }

    // Resize to 384x384
    std::vector<unsigned char> resized =
        resizeNearest(img, imgW, imgH, 3, INPUT_WIDTH, INPUT_HEIGHT);
    stbi_image_free(img);

    // HWC -> CHW, normalize: (pixel/255.0 - mean) / std
    int pixels = INPUT_HEIGHT * INPUT_WIDTH;
    tensorData.resize(pixels * INPUT_CHANNELS);
    for (int y = 0; y < INPUT_HEIGHT; y++) {
        for (int x = 0; x < INPUT_WIDTH; x++) {
            int hwc = (y * INPUT_WIDTH + x) * 3;
            for (int c = 0; c < 3; c++) {
                float pixel = static_cast<float>(resized[hwc + c]) / 255.0f;
                tensorData[c * pixels + y * INPUT_WIDTH + x] =
                    (pixel - MEAN[c]) / STD[c];
            }
        }
    }
}

// ========================================================================
// Postprocessing
// ========================================================================

// Apply softmax to logits for probability distribution
static void softmax(float* data, int size)
{
    float maxVal = *std::max_element(data, data + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = std::exp(data[i] - maxVal);
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

// Structure for top-k results
struct TopKResult {
    int index;
    float score;
};

// Print top-k predictions
static void printTopK(float* logits, int numClasses, int k)
{
    // Create index array and sort by logit value (descending)
    std::vector<int> indices(numClasses);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [logits](int a, int b) { return logits[a] > logits[b]; });

    fprintf(stdout, "Top-%d predictions:\n", k);
    for (int i = 0; i < k; i++) {
        fprintf(stdout, "  [%d] class=%d  score=%.4f\n",
            i, indices[i], logits[indices[i]]);
    }
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

    fprintf(stdout, "ViT-Base/32 Runner\n");
    fprintf(stdout, "Target input: %dx%d (HxW)\n", INPUT_HEIGHT, INPUT_WIDTH);
    fprintf(stdout, "Model: %s\n", modelPath);
    fprintf(stdout, "Image: %s\n", imagePath);
    fprintf(stdout, "Iterations: %d (1 warm-up + %d measured)\n",
            iterations, iterations - 1);

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vit_runner");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    fprintf(stdout, "Loading model...\n");
    Ort::Session session(env, modelPath, sessionOpts);

    // Get input/output names
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNamePtrs = {outputName.get()};

    fprintf(stdout, "Input:  %s\n", inputName.get());
    fprintf(stdout, "Output: %s\n", outputName.get());

    // Preprocess image
    fprintf(stdout, "Preprocessing image...\n");
    std::vector<float> tensorData;
    std::vector<int64_t> inputShapeVec;
    preprocess(imagePath, tensorData, inputShapeVec);

    // Use static array to avoid stack corruption by ORT Session constructor
    // (ORT Session uses large stack frames that can corrupt local variables under QEMU)
    static int64_t inputShape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
    static const size_t inputShapeLen = 4;

    fprintf(stdout, "Input shape: [1, %d, %d, %d], %zu elements\n",
        INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, tensorData.size());

    // Create input tensor
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(),
        inputShape, inputShapeLen);

    // Run inference
    fprintf(stdout, "Running inference...\n");
    const char* inputNames[] = {inputName.get()};

    Ort::Value lastOutput;

    for (int i = 0; i < iterations; i++) {
        fprintf(stdout, "  [%d/%d]%s\n", i + 1, iterations,
                i == 0 ? " (warm-up)" : "");
        auto outputs = session.Run(
            Ort::RunOptions{},
            inputNames, &inputTensor, 1,
            outputNamePtrs.data(), outputNamePtrs.size());

        if (i == iterations - 1) {
            lastOutput = std::move(outputs[0]);
        }
    }

    // Process output: logits [1, 1000]
    auto outputInfo = lastOutput.GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();
    float* logits = lastOutput.GetTensorMutableData<float>();
    int64_t totalElements = outputInfo.GetElementCount();

    fprintf(stdout, "\nOutput shape: [");
    for (size_t d = 0; d < outputShape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(outputShape[d]));
    }
    fprintf(stdout, "]\n");
    fprintf(stdout, "Total elements: %lld\n", static_cast<long long>(totalElements));

    // Apply softmax for probabilities
    softmax(logits, static_cast<int>(totalElements));

    // Print top-5 predictions
    printTopK(logits, static_cast<int>(totalElements), 5);

    // Validation summary
    fprintf(stdout, "\n--- Validation ---\n");
    float maxProb = *std::max_element(logits, logits + totalElements);
    int maxIdx = static_cast<int>(std::max_element(logits, logits + totalElements) - logits);
    fprintf(stdout, "Top class: %d (score=%.4f)\n", maxIdx, maxProb);
    fprintf(stdout, "Sum of probabilities: %.6f (expected ~1.0: %s)\n",
        std::accumulate(logits, logits + totalElements, 0.0f),
        std::abs(std::accumulate(logits, logits + totalElements, 0.0f) - 1.0f) < 0.001f ? "PASS" : "FAIL");

    fprintf(stdout, "Done.\n");
    return 0;
}

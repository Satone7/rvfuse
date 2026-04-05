#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static constexpr int MODEL_INPUT_SIZE = 640;

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

static void preprocess(
    const char* imagePath,
    std::vector<float>& tensorData,
    std::vector<int64_t>& inputShape)
{
    int imgW, imgH, imgC;
    unsigned char* img = stbi_load(imagePath, &imgW, &imgH, &imgC, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load image %s\n", imagePath);
        exit(1);
    }

    std::vector<unsigned char> resized =
        resizeNearest(img, imgW, imgH, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    stbi_image_free(img);

    // HWC -> CHW, normalize to [0, 1]
    int pixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
    tensorData.resize(pixels * 3);
    for (int y = 0; y < MODEL_INPUT_SIZE; y++) {
        for (int x = 0; x < MODEL_INPUT_SIZE; x++) {
            int hwc = (y * MODEL_INPUT_SIZE + x) * 3;
            tensorData[0 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 0] / 255.0f;
            tensorData[1 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 1] / 255.0f;
            tensorData[2 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 2] / 255.0f;
        }
    }

    inputShape = {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE};
}

static void printTopDetections(Ort::Value& tensor, int showCount)
{
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    float* data = tensor.GetTensorMutableData<float>();
    int64_t total = info.GetElementCount();

    fprintf(stdout, "Output shape: [");
    for (size_t d = 0; d < shape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(shape[d]));
    }
    fprintf(stdout, "]\n");

    // YOLO output: [numDets, 6] where cols = x1, y1, x2, y2, conf, cls
    int cols = 6;
    int numDets = static_cast<int>(total / cols);
    int show = std::min(numDets, showCount);
    fprintf(stdout, "Top %d detections:\n", show);
    for (int d = 0; d < show; d++) {
        int i = d * cols;
        fprintf(stdout, "  [%d] box=(%.1f,%.1f,%.1f,%.1f) conf=%.3f cls=%.0f\n",
            d, data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5]);
    }
}

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

    fprintf(stdout, "Model: %s\n", modelPath);
    fprintf(stdout, "Image: %s\n", imagePath);
    fprintf(stdout, "Iterations: %d (1 warm-up + %d measured)\n",
            iterations, iterations - 1);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_runner");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    fprintf(stdout, "Loading model...\n");
    Ort::Session session(env, modelPath, sessionOpts);

    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNamePtrs;
    outputNamePtrs.push_back(outputName.get());

    fprintf(stdout, "Preprocessing image...\n");
    std::vector<float> tensorData;
    std::vector<int64_t> inputShape;
    preprocess(imagePath, tensorData, inputShape);

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(),
        inputShape.data(), inputShape.size());

    fprintf(stdout, "Running inference...\n");
    const char* inputNames[] = {inputName.get()};

    for (int i = 0; i < iterations; i++) {
        fprintf(stdout, "  [%d/%d]%s\n", i + 1, iterations,
                i == 0 ? " (warm-up)" : "");
        auto outputs = session.Run(
            Ort::RunOptions{},
            inputNames, &inputTensor, 1,
            outputNamePtrs.data(), outputNamePtrs.size());

        if (i == iterations - 1) {
            printTopDetections(outputs[0], 5);
        }
    }

    fprintf(stdout, "Done.\n");
    return 0;
}

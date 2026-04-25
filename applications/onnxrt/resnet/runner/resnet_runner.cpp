// resnet_runner.cpp — ResNet ONNX Runtime inference runner for RVFuse profiling.
// Implements ImageNet preprocessing (resize, normalize) and inference.
// Usage: ./resnet_runner <model.onnx> [iterations] [--image <image.jpg>]

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "onnxruntime_cxx_api.h"

// ImageNet normalization constants
constexpr float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[]  = {0.229f, 0.224f, 0.225f};

// Resize image to target dimensions using bilinear interpolation
void resize_image(const uint8_t* src, int src_w, int src_h, int src_c,
                  uint8_t* dst, int dst_w, int dst_h) {
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float sx = static_cast<float>(x) * src_w / dst_w;
            float sy = static_cast<float>(y) * src_h / dst_h;
            int x0 = static_cast<int>(sx);
            int y0 = static_cast<int>(sy);
            int x1 = std::min(x0 + 1, src_w - 1);
            int y1 = std::min(y0 + 1, src_h - 1);
            float dx = sx - x0;
            float dy = sy - y0;

            for (int c = 0; c < src_c; c++) {
                float v00 = src[(y0 * src_w + x0) * src_c + c];
                float v01 = src[(y0 * src_w + x1) * src_c + c];
                float v10 = src[(y1 * src_w + x0) * src_c + c];
                float v11 = src[(y1 * src_w + x1) * src_c + c];
                float v = v00 * (1 - dx) * (1 - dy) +
                          v01 * dx * (1 - dy) +
                          v10 * (1 - dx) * dy +
                          v11 * dx * dy;
                dst[(y * dst_w + x) * src_c + c] = static_cast<uint8_t>(v);
            }
        }
    }
}

// Preprocess image for ResNet: resize to 224x224, normalize with ImageNet stats
void preprocess_image(const uint8_t* img_data, int img_w, int img_h, int img_c,
                      float* input_data, int target_w = 224, int target_h = 224) {
    // Resize if needed
    std::vector<uint8_t> resized;
    const uint8_t* src = img_data;
    int src_w = img_w, src_h = img_h;

    if (img_w != target_w || img_h != target_h) {
        resized.resize(target_w * target_h * img_c);
        resize_image(img_data, img_w, img_h, img_c, resized.data(), target_w, target_h);
        src = resized.data();
        src_w = target_w;
        src_h = target_h;
    }

    // Convert to float and normalize (channel-first format for ResNet)
    // ResNet expects: N x C x H x W with normalized values
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < target_h; y++) {
            for (int x = 0; x < target_w; x++) {
                uint8_t pixel = src[(y * target_w + x) * 3 + c];
                input_data[c * target_h * target_w + y * target_w + x] =
                    (static_cast<float>(pixel) / 255.0f - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
}

// Generate random ImageNet-style input (for profiling without real images)
void generate_random_input(float* data, size_t size, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations] [--image <image.jpg>]" << std::endl;
        std::cerr << "  iterations: number of inference runs (default: 10)" << std::endl;
        std::cerr << "  --image: optional input image for realistic inference" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    int iterations = 10;
    const char* image_path = nullptr;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (atoi(argv[i]) > 0) {
            iterations = atoi(argv[i]);
        }
    }

    // Initialize ORT
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet-runner");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path, session_options);

    // Read input info from model
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();

    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();

    // Handle dynamic dimensions
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] < 0) input_shape[i] = 1;
    }

    // Print input info
    std::cout << "Input: " << input_name << " shape=[";
    size_t total_elements = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ",";
        total_elements *= static_cast<size_t>(input_shape[i]);
    }
    std::cout << "] type=" << static_cast<int>(input_type) << std::endl;

    // Validate input type
    if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "Error: ResNet expects float32 input, got type "
                  << static_cast<int>(input_type) << std::endl;
        return 1;
    }

    // Print output info
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();
    std::cout << "Output: " << output_name << std::endl;

    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input_data(total_elements);

    if (image_path) {
        // Load and preprocess image
        int img_w, img_h, img_c;
        uint8_t* img_data = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
        if (!img_data) {
            std::cerr << "Error: failed to load image: " << image_path << std::endl;
            return 1;
        }
        std::cout << "Loaded image: " << img_w << "x" << img_h << " channels=" << img_c << std::endl;

        // Preprocess for ResNet
        preprocess_image(img_data, img_w, img_h, 3, input_data.data());
        stbi_image_free(img_data);
        std::cout << "Preprocessed to 224x224 normalized tensor" << std::endl;
    } else {
        // Generate random input for profiling
        generate_random_input(input_data.data(), total_elements);
        std::cout << "Generated random input (" << total_elements << " elements)" << std::endl;
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), total_elements,
        input_shape.data(), input_shape.size());

    // Warm-up run
    std::cout << "Warm-up run..." << std::endl;
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    try {
        auto warmup_output = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);
        std::cout << "Warm-up OK" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Warm-up failed: " << e.what() << std::endl;
        return 1;
    }

    // Measured runs
    std::cout << "Running " << iterations << " iterations..." << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto output = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "Done: " << iterations << " iterations in " << total_ms << " ms" << std::endl;
    std::cout << "Average: " << total_ms / iterations << " ms/iter" << std::endl;

    return 0;
}
// generic_ort_runner.cpp — Generic ONNX Runtime inference runner for perf profiling.
// Reads input shape from the model, generates random test data, runs inference.
// Usage: ./generic_ort_runner <model.onnx> [iterations]

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cstdlib>

#include "onnxruntime_cxx_api.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [iterations=10]" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    int iterations = argc >= 3 ? std::atoi(argv[2]) : 10;
    if (iterations <= 0) iterations = 10;

    // Initialize ORT
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort-perf");
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

    // Print input info
    std::cout << "Input: " << input_name << " shape=[";
    size_t total_elements = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
        // Handle dynamic dimensions (negative values)
        if (input_shape[i] < 0) input_shape[i] = 1;
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ",";
        total_elements *= static_cast<size_t>(input_shape[i]);
    }
    std::cout << "] type=" << static_cast<int>(input_type) << std::endl;

    // Print output info
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();
    std::cout << "Output: " << output_name << std::endl;

    // Generate random input data
    std::vector<float> input_data(total_elements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : input_data) {
        v = dist(gen);
    }

    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
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

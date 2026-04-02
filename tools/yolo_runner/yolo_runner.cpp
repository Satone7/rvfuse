// YOLO Inference Runner
// Loads a YOLO ONNX model, preprocesses an image, runs inference,
// applies NMS, and prints detected objects.
//
// Build: via CMakeLists.txt with ONNX Runtime and stb_image
// Usage: yolo_inference <model.onnx> <image.jpg>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int INPUT_WIDTH    = 640;
static constexpr int INPUT_HEIGHT   = 640;
static constexpr int INPUT_CHANNELS = 3;

static constexpr float NMS_THRESHOLD  = 0.45f;
static constexpr float CONF_THRESHOLD = 0.25f;

// YOLO11n output shape: [1, 84, 8400]
// 84 = 4 bbox coords (cx, cy, w, h) + 80 class scores
static constexpr int NUM_CLASSES   = 80;
static constexpr int BBOX_PARAMS   = 4;
static constexpr int OUTPUT_ROWS   = BBOX_PARAMS + NUM_CLASSES; // 84
static constexpr int NUM_DETECTIONS = 8400;

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

// Preprocess an image: bilinear resize to INPUT_WIDTH x INPUT_HEIGHT,
// convert HWC to CHW, and normalize pixel values to [0, 1].
//
// Parameters:
//   img_data  - raw pixel data from stb_image (HWC, uint8)
//   src_w     - source image width
//   src_h     - source image height
//   src_channels - source image channels (must be 3)
//
// Returns:
//   Vector of float with size INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH
//   in CHW order, normalized to [0, 1].
static std::vector<float> preprocess_image(const unsigned char* img_data,
                                           int src_w, int src_h,
                                           int src_channels) {
    const int dst_w = INPUT_WIDTH;
    const int dst_h = INPUT_HEIGHT;
    const int channels = INPUT_CHANNELS;

    std::vector<float> output(channels * dst_h * dst_w);

    // Bilinear resize + HWC->CHW + normalize
    for (int c = 0; c < channels; ++c) {
        for (int dy = 0; dy < dst_h; ++dy) {
            // Map destination pixel center back to source coordinates
            float src_y = (static_cast<float>(dy) + 0.5f) *
                          (static_cast<float>(src_h) / static_cast<float>(dst_h)) -
                          0.5f;
            int y0 = static_cast<int>(std::floor(src_y));
            int y1 = y0 + 1;
            float fy = src_y - static_cast<float>(y0);
            // Clamp
            y0 = std::max(0, std::min(y0, src_h - 1));
            y1 = std::max(0, std::min(y1, src_h - 1));

            for (int dx = 0; dx < dst_w; ++dx) {
                float src_x = (static_cast<float>(dx) + 0.5f) *
                              (static_cast<float>(src_w) / static_cast<float>(dst_w)) -
                              0.5f;
                int x0 = static_cast<int>(std::floor(src_x));
                int x1 = x0 + 1;
                float fx = src_x - static_cast<float>(x0);
                // Clamp
                x0 = std::max(0, std::min(x0, src_w - 1));
                x1 = std::max(0, std::min(x1, src_w - 1));

                // Bilinear interpolation for channel c
                float v00 = static_cast<float>(
                    img_data[(y0 * src_w + x0) * src_channels + c]);
                float v01 = static_cast<float>(
                    img_data[(y0 * src_w + x1) * src_channels + c]);
                float v10 = static_cast<float>(
                    img_data[(y1 * src_w + x0) * src_channels + c]);
                float v11 = static_cast<float>(
                    img_data[(y1 * src_w + x1) * src_channels + c]);

                float val = v00 * (1.0f - fx) * (1.0f - fy) +
                            v01 * fx * (1.0f - fy) +
                            v10 * (1.0f - fx) * fy +
                            v11 * fx * fy;

                // Normalize to [0, 1]
                val /= 255.0f;

                // CHW layout: channel-major, then row, then column
                output[c * dst_h * dst_w + dy * dst_w + dx] = val;
            }
        }
    }

    return output;
}

// ---------------------------------------------------------------------------
// Detection structure
// ---------------------------------------------------------------------------

struct Detection {
    float x;          // bounding box center x (in input image coords)
    float y;          // bounding box center y (in input image coords)
    float w;          // bounding box width
    float h;          // bounding box height
    float confidence; // max class confidence * objectness
    int class_id;     // argmax class index
};

// ---------------------------------------------------------------------------
// Non-Maximum Suppression
// ---------------------------------------------------------------------------

// Compute simplified IoU between two detections using center/width/height.
static float compute_iou(const Detection& a, const Detection& b) {
    float ax1 = a.x - a.w / 2.0f;
    float ay1 = a.y - a.h / 2.0f;
    float ax2 = a.x + a.w / 2.0f;
    float ay2 = a.y + a.h / 2.0f;

    float bx1 = b.x - b.w / 2.0f;
    float by1 = b.y - b.h / 2.0f;
    float bx2 = b.x + b.w / 2.0f;
    float by2 = b.y + b.h / 2.0f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// Parse YOLO11n output tensor [1, 84, 8400], filter by confidence,
// and apply NMS.
//
// The output layout after the model's internal transpose is column-major
// w.r.t. the [84, 8400] matrix: row i holds all 8400 values for that
// attribute (first 4 rows = bbox cx,cy,w,h; remaining 80 = class scores).
// In memory: output[i * 8400 + j] gives attribute i for detection j.
static std::vector<Detection> apply_nms(const float* output_data,
                                        int output_rows, int num_detections) {
    std::vector<Detection> candidates;

    // For each detection, find the class with max score
    for (int d = 0; d < num_detections; ++d) {
        // Find best class score among the 80 class rows (rows 4..83)
        int best_class = 0;
        float best_score = output_data[BBOX_PARAMS * num_detections + d];

        for (int c = 1; c < NUM_CLASSES; ++c) {
            float score = output_data[(BBOX_PARAMS + c) * num_detections + d];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        // Filter by confidence threshold
        if (best_score < CONF_THRESHOLD) {
            continue;
        }

        Detection det;
        det.x = output_data[0 * num_detections + d]; // cx
        det.y = output_data[1 * num_detections + d]; // cy
        det.w = output_data[2 * num_detections + d]; // width
        det.h = output_data[3 * num_detections + d]; // height
        det.confidence = best_score;
        det.class_id = best_class;

        candidates.push_back(det);
    }

    // Sort candidates by confidence descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // NMS: greedily keep highest-confidence detections, suppress overlapping
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> results;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            float iou = compute_iou(candidates[i], candidates[j]);
            if (iou > NMS_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <model.onnx> <image.jpg>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    // ------------------------------------------------------------------
    // 1. Load and preprocess image
    // ------------------------------------------------------------------
    int img_w = 0, img_h = 0, img_channels = 0;
    unsigned char* img_data = stbi_load(image_path, &img_w, &img_h,
                                        &img_channels, INPUT_CHANNELS);
    if (!img_data) {
        std::fprintf(stderr, "Error: failed to load image '%s'\n", image_path);
        return 1;
    }

    std::fprintf(stdout, "Loaded image: %dx%d (%d channels)\n",
                 img_w, img_h, INPUT_CHANNELS);

    std::vector<float> input_tensor = preprocess_image(
        img_data, img_w, img_h, INPUT_CHANNELS);
    stbi_image_free(img_data);

    // ------------------------------------------------------------------
    // 2. Initialize ONNX Runtime environment and load model
    // ------------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo-inference");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path, session_options);

    // ------------------------------------------------------------------
    // 3. Prepare input tensor
    // ------------------------------------------------------------------
    // Query input node info
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();

    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();

    std::vector<int64_t> input_dims = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
    size_t input_tensor_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;

    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor_size,
        input_dims.data(), input_dims.size());

    // ------------------------------------------------------------------
    // 4. Run inference
    // ------------------------------------------------------------------
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_alloc.get();

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    std::fprintf(stdout, "Running inference...\n");

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor_ort, 1,
        output_names, 1);

    // ------------------------------------------------------------------
    // 5. Post-process: parse output, apply NMS
    // ------------------------------------------------------------------
    Ort::Value& output_tensor = output_tensors[0];
    auto output_type_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_type_info.GetShape();

    // Expected shape: [1, 84, 8400]
    std::fprintf(stdout, "Output shape: [%lld, %lld, %lld]\n",
                 static_cast<long long>(output_shape[0]),
                 static_cast<long long>(output_shape[1]),
                 static_cast<long long>(output_shape[2]));

    const float* output_data = output_tensor.GetTensorData<float>();

    int out_rows = static_cast<int>(output_shape[1]);
    int out_cols = static_cast<int>(output_shape[2]);

    std::vector<Detection> detections = apply_nms(output_data, out_rows, out_cols);

    // ------------------------------------------------------------------
    // 6. Print results
    // ------------------------------------------------------------------
    std::fprintf(stdout, "\nDetections: %zu\n", detections.size());
    std::fprintf(stdout, "%-8s %-12s %-10s %-10s %-10s %-10s\n",
                 "Class", "Confidence", "cx", "cy", "w", "h");
    std::fprintf(stdout, "------  -----------  ----------  ----------  "
                         "----------  ----------\n");

    for (const auto& det : detections) {
        std::fprintf(stdout, "%-8d %-12.4f %-10.2f %-10.2f %-10.2f %-10.2f\n",
                     det.class_id, det.confidence,
                     det.x, det.y, det.w, det.h);
    }

    return 0;
}

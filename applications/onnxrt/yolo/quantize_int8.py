#!/usr/bin/env python3
"""Quantize YOLO11n ONNX model to INT8 using ORT dynamic quantization."""

import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    model_fp32 = os.path.join(project_root, "output", "yolo11n.onnx")
    model_int8 = os.path.join(project_root, "output", "yolo11n_int8.onnx")

    if not os.path.exists(model_fp32):
        print(f"FP32 model not found: {model_fp32}", file=sys.stderr)
        print("Run prepare_model.sh first.", file=sys.stderr)
        sys.exit(1)

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing {model_fp32} -> {model_int8} ...")
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_int8,
        weight_type=QuantType.QUInt8,
        per_channel=True,
    )
    print(f"INT8 model saved: {model_int8}")

    # Verify
    import onnx
    model = onnx.load(model_int8)
    qnodes = [n for n in model.graph.node
              if 'QuantizeLinear' in n.op_type
              or 'DequantizeLinear' in n.op_type
              or 'QLinear' in n.op_type]
    print(f"Q/DQ/QLinear nodes: {len(qnodes)}")
    for n in qnodes[:10]:
        print(f"  {n.op_type}: {n.name or '(unnamed)'}")

    # Generate ORT format
    import onnxruntime as ort
    model_ort = os.path.join(project_root, "output", "yolo11n_int8.ort")
    so = ort.SessionOptions()
    so.optimized_model_filepath = model_ort
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort.InferenceSession(model_int8, so)
    print(f"ORT format saved: {model_ort}")

if __name__ == "__main__":
    main()

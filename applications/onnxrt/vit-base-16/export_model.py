#!/usr/bin/env python3
"""Export ViT-Base/16 from HuggingFace to ONNX."""

import torch
from transformers import ViTForImageClassification

MODEL_ID = "google/vit-base-patch16-224"
OUTPUT_PATH = "applications/onnxrt/vit-base-16/model/vit_base_patch16_224.onnx"

print(f"Loading {MODEL_ID} ...")
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=17,
)

print(f"Exported to {OUTPUT_PATH}")

# Smoke test: run inference and check output shape
import onnxruntime as ort
session = ort.InferenceSession(OUTPUT_PATH)
outputs = session.run(None, {"pixel_values": dummy_input.numpy()})
print(f"ONNX smoke test: output shape={outputs[0].shape}, top-5={outputs[0][0].argsort()[-5:][::-1]}")

# Also save to shared location
import os, shutil
shared = "output/models/vit_base_patch16_224.onnx"
os.makedirs("output/models", exist_ok=True)
shutil.copy(OUTPUT_PATH, shared)
print(f"Copied to {shared}")

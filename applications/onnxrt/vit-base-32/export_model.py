#!/usr/bin/env python3
"""Export ViT-Base/32 from HuggingFace to ONNX.

Note: google/vit-base-patch32-224 does not exist. Using google/vit-base-patch32-384
(384x384 input, 12x12=144 patches + CLS = 145 tokens vs 197 for ViT-Base/16@224).
"""

import torch
from transformers import ViTForImageClassification

MODEL_ID = "google/vit-base-patch32-384"
OUTPUT_PATH = "applications/onnxrt/vit-base-32/model/vit_base_patch32_384.onnx"

print(f"Loading {MODEL_ID} ...")
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()

dummy_input = torch.randn(1, 3, 384, 384)

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

# Smoke test
import onnxruntime as ort
session = ort.InferenceSession(OUTPUT_PATH)
outputs = session.run(None, {"pixel_values": dummy_input.numpy()})
print(f"ONNX smoke test: output shape={outputs[0].shape}, top-5={outputs[0][0].argsort()[-5:][::-1]}")

import os, shutil
shared = "output/models/vit_base_patch32_384.onnx"
os.makedirs("output/models", exist_ok=True)
shutil.copy(OUTPUT_PATH, shared)
print(f"Copied to {shared}")

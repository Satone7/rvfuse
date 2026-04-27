#!/usr/bin/env python3
"""Export SuperPoint from Magic Leap pretrained weights to ONNX.

Clones the SuperPointPretrainedNetwork repo if not already present,
loads the pretrained model, and exports to ONNX.
"""

import os, sys, shutil
import torch

REPO_URL = "https://github.com/magicleap/SuperPointPretrainedNetwork"
REF_DIR = "applications/onnxrt/superpoint/reference"
OUTPUT_PATH = "applications/onnxrt/superpoint/model/superpoint.onnx"
WEIGHTS_PATH = os.path.join(REF_DIR, "superpoint_v1.pth")

# Clone reference repo if needed
if not os.path.isdir(os.path.join(REF_DIR, ".git")):
    print(f"Cloning {REPO_URL} ...")
    os.makedirs(REF_DIR, exist_ok=True)
    os.system(f"git clone --depth 1 {REPO_URL} {REF_DIR}")
else:
    print("Reference repo already cloned.")

sys.path.insert(0, REF_DIR)
from demo_superpoint import SuperPointNet

print(f"Loading pretrained model from {WEIGHTS_PATH} ...")
model = SuperPointNet()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# SuperPoint input: grayscale image (B, 1, H, W)
# Use 480x640 (VGA) for reproducible BBV profiling
dummy_input = torch.randn(1, 1, 480, 640)

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    input_names=["image"],
    output_names=["semi", "desc"],
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
)

print(f"Exported to {OUTPUT_PATH}")

# Smoke test
import onnxruntime as ort
session = ort.InferenceSession(OUTPUT_PATH)
outputs = session.run(None, {"image": dummy_input.numpy()})
semi, desc = outputs
print(f"ONNX smoke test: semi shape={semi.shape}, desc shape={desc.shape}")
print(f"  semi: min={semi.min():.4f}, max={semi.max():.4f}")
print(f"  desc: L2 norm mean={torch.norm(torch.tensor(desc), dim=1).mean():.4f}")

shared = "output/models/superpoint.onnx"
os.makedirs("output/models", exist_ok=True)
shutil.copy(OUTPUT_PATH, shared)
print(f"Copied to {shared}")

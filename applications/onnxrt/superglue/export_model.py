#!/usr/bin/env python3
"""Export SuperGlue GNN (without Sinkhorn) from Magic Leap pretrained weights to ONNX.

Exports only the keypoint encoder + GNN layers + final projection.
Sinkhorn optimal transport is implemented in the C++ runner.
"""

import os, sys, shutil
import torch
import torch.nn as nn

REPO_URL = "https://github.com/magicleap/SuperGluePretrainedNetwork"
REF_DIR = "applications/onnxrt/superglue/reference"
OUTPUT_PATH = "applications/onnxrt/superglue/model/superglue_gnn.onnx"
WEIGHTS_PATH = os.path.join(REF_DIR, "models/weights/superglue_outdoor.pth")

# Clone reference repo if needed
if not os.path.isdir(os.path.join(REF_DIR, ".git")):
    print(f"Cloning {REPO_URL} ...")
    os.makedirs(REF_DIR, exist_ok=True)
    os.system(f"git clone --depth 1 {REPO_URL} {REF_DIR}")
else:
    print("Reference repo already cloned.")

sys.path.insert(0, os.path.join(REF_DIR, "models"))
from superglue import SuperGlue

# Load full SuperGlue model
full_config = {
    "weights": "outdoor",
    "sinkhorn_iterations": 100,
    "match_threshold": 0.2,
}
print(f"Loading pretrained model ...")
full_model = SuperGlue(full_config)
ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
full_model.load_state_dict(ckpt)
full_model.eval()

class SuperGlueGNNONNX(nn.Module):
    """GNN-only SuperGlue for ONNX export (no Sinkhorn)."""

    def __init__(self, sg_model):
        super().__init__()
        self.kenc = sg_model.kenc
        self.gnn = sg_model.gnn
        self.final_proj = sg_model.final_proj

    def forward(self, kpts0, scores0, desc0, kpts1, scores1, desc1):
        # Keypoint encoding
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # GNN layers (9 alternating self/cross attention)
        for layer_idx, layer in enumerate(self.gnn.layers):
            name = self.gnn.names[layer_idx]
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

        # Final projection: pairwise matching scores
        mdesc0 = self.final_proj(desc0)
        mdesc1 = self.final_proj(desc1)
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        return scores / mdesc0.shape[1] ** 0.5

onnx_model = SuperGlueGNNONNX(full_model)
onnx_model.eval()

# Fixed N_max=1024 for reproducible BBV profiling
# Conv1d-based model expects (B, D, N) descriptor format
N = 1024
dummy_kpts0 = torch.randn(1, N, 2)
dummy_scores0 = torch.rand(1, N)
dummy_desc0 = torch.randn(1, 256, N)
dummy_kpts1 = torch.randn(1, N, 2)
dummy_scores1 = torch.rand(1, N)
dummy_desc1 = torch.randn(1, 256, N)

torch.onnx.export(
    onnx_model,
    (dummy_kpts0, dummy_scores0, dummy_desc0, dummy_kpts1, dummy_scores1, dummy_desc1),
    OUTPUT_PATH,
    input_names=["kpts0", "scores0", "desc0", "kpts1", "scores1", "desc1"],
    output_names=["scores_matrix"],
    opset_version=17,
)

print(f"Exported to {OUTPUT_PATH}")

# Smoke test with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession(OUTPUT_PATH)
inputs = {
    "kpts0": dummy_kpts0.numpy(),
    "scores0": dummy_scores0.numpy(),
    "desc0": dummy_desc0.numpy(),
    "kpts1": dummy_kpts1.numpy(),
    "scores1": dummy_scores1.numpy(),
    "desc1": dummy_desc1.numpy(),
}
outputs = session.run(None, inputs)
scores_matrix = outputs[0]
print(f"ONNX smoke test: scores_matrix shape={scores_matrix.shape}")
print(f"  range=[{scores_matrix.min():.4f}, {scores_matrix.max():.4f}]")

shared = "output/models/superglue_gnn.onnx"
os.makedirs("output/models", exist_ok=True)
shutil.copy(OUTPUT_PATH, shared)
print(f"Copied to {shared}")

#!/usr/bin/env python3
"""
Export DeepLabV3-MobileNetV3 to ONNX for use with OpenCV DNN.

Requirements:
    pip install torch torchvision

Usage:
    python3 export_model.py

Creates ../../data/models/deeplabv3/deeplabv3_mobilenetv3.onnx and voc.names.

The torchvision model returns a dictionary {"out": ..., "aux": ...}, which
ONNX cannot express, so it is wrapped to return only the segmentation map.
The input size is fixed at 384x384 on export: OpenCV feeds it exactly that.

Author: José Miguel Guerrero Hernández
"""

import os
import torch
import torchvision

MODEL_DIR = "../../data/models/deeplabv3"
INPUT_SIZE = 384

# The 21 classes of PASCAL VOC, in the order the network scores them
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class OnlyOutput(torch.nn.Module):
    """Keeps the segmentation map and drops the auxiliary head."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(MODEL_DIR, "deeplabv3_mobilenetv3.onnx")

    print("=== Exporting DeepLabV3-MobileNetV3 to ONNX ===")
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights="DEFAULT").eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(OnlyOutput(model), dummy, onnx_path, opset_version=12,
                      input_names=["input"], output_names=["output"])
    print(f"Model saved to {onnx_path}")

    names_path = os.path.join(MODEL_DIR, "voc.names")
    with open(names_path, "w") as f:
        f.write("\n".join(VOC_CLASSES) + "\n")
    print(f"Class names saved to {names_path}")

    print("\n=== Done! ===")
    for name in sorted(os.listdir(MODEL_DIR)):
        size_mb = os.path.getsize(os.path.join(MODEL_DIR, name)) / (1024 * 1024)
        print(f"  {name} ({size_mb:.1f} MB)")
    print("\nNow build and run:")
    print("  make")
    print("  ./semantic_segmentation ../../data/futbol.png")


if __name__ == "__main__":
    main()

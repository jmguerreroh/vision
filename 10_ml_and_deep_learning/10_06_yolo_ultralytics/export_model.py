#!/usr/bin/env python3
"""
Export YOLO11 model to ONNX format for use with OpenCV DNN.

Requirements:
    pip install ultralytics

Usage:
    python3 export_model.py

This will create cfg/yolo11n.onnx and cfg/coco.names.

IMPORTANT: Requires OpenCV >= 4.9 to load the exported ONNX model.

Author: José Miguel Guerrero Hernández
"""

import os
from ultralytics import YOLO

CFG_DIR = "../../data/models/yolo11"
os.makedirs(CFG_DIR, exist_ok=True)

# YOLO11 nano — smallest and fastest model (~10 MB)
MODEL = "yolo11n.pt"
ONNX_NAME = MODEL.replace(".pt", ".onnx")

print(f"=== Exporting {MODEL} to ONNX ===")
model = YOLO(MODEL)
model.export(format="onnx", imgsz=640, simplify=True)

# Move the exported file to cfg/
onnx_src = ONNX_NAME
onnx_dst = os.path.join(CFG_DIR, ONNX_NAME)
if os.path.exists(onnx_src):
    os.rename(onnx_src, onnx_dst)
    print(f"Model saved to {onnx_dst}")

# Clean up the .pt file
if os.path.exists(MODEL):
    os.remove(MODEL)
    print(f"Cleaned up {MODEL}")

# Create COCO class names file
COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

names_path = os.path.join(CFG_DIR, "coco.names")
with open(names_path, "w") as f:
    f.write("\n".join(COCO_NAMES) + "\n")
print(f"Class names saved to {names_path}")

print(f"\n=== Done! ===")
print(f"Files in {CFG_DIR}/:")
for fname in sorted(os.listdir(CFG_DIR)):
    fpath = os.path.join(CFG_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {fname} ({size_mb:.1f} MB)")
print("\nNow build and run:")
print("  make")
print("  ./yolo11 ../../data/vtest.avi")

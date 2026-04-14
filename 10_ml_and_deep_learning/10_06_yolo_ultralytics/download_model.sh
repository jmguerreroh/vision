#!/bin/bash
# Download/export YOLO11 model for OpenCV DNN
# Author: José Miguel Guerrero Hernández
#
# Requires: pip install ultralytics
# NOTE: ONNX model requires OpenCV >= 4.9

set -e

MODEL_DIR="../../data/models/yolo11"
ONNX_FILE="$MODEL_DIR/yolo11n.onnx"
NAMES_FILE="$MODEL_DIR/coco.names"

# Skip if model already exists
if [ -f "$ONNX_FILE" ] && [ -f "$NAMES_FILE" ]; then
  echo "YOLO11 model already exists, skipping."
  exit 0
fi

mkdir -p "$MODEL_DIR"

echo "=== Exporting YOLO11n to ONNX ==="
python3 export_model.py

echo ""
echo "=== Done! ==="
echo "Files in $MODEL_DIR/:"
ls -lh "$MODEL_DIR/"

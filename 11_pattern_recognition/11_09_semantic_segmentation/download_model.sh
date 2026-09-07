#!/bin/bash
# Export the DeepLabV3 model used by the semantic segmentation example
# Author: José Miguel Guerrero Hernández
#
# Requires: pip install torch torchvision

set -e

MODEL_DIR="../../data/models/deeplabv3"
ONNX_FILE="$MODEL_DIR/deeplabv3_mobilenetv3.onnx"
NAMES_FILE="$MODEL_DIR/voc.names"

# Skip if the model is already there
if [ -f "$ONNX_FILE" ] && [ -f "$NAMES_FILE" ]; then
  echo "DeepLabV3 model already exists, skipping."
  exit 0
fi

mkdir -p "$MODEL_DIR"

echo "=== Exporting DeepLabV3-MobileNetV3 to ONNX ==="
python3 export_model.py

echo ""
echo "=== Done! ==="
ls -lh "$MODEL_DIR/"

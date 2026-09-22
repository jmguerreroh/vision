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

PIP_PACKAGES="torch torchvision onnxscript"
MISSING=""
for mod in torch torchvision onnxscript; do
  python3 -c "import $mod" 2>/dev/null || MISSING="$MISSING $mod"
done

if [ -n "$MISSING" ]; then
  echo "WARNING: missing Python modules:$MISSING"
  if [ -t 0 ]; then
    read -r -p "Install them now with 'pip install --user --break-system-packages $PIP_PACKAGES'? [y/N] " reply
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      pip install --user --break-system-packages $PIP_PACKAGES
      MISSING=""
      for mod in torch torchvision onnxscript; do
        python3 -c "import $mod" 2>/dev/null || MISSING="$MISSING $mod"
      done
    fi
  fi
fi

if [ -n "$MISSING" ]; then
  echo "Skipping DeepLabV3 export - still missing:$MISSING"
  echo "To install manually: pip install --user --break-system-packages $PIP_PACKAGES"
  exit 0
fi

echo "=== Exporting DeepLabV3-MobileNetV3 to ONNX ==="
python3 export_model.py

echo ""
echo "=== Done! ==="
ls -lh "$MODEL_DIR/"

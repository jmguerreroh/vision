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

# Check for the Python modules needed to export to ONNX; ultralytics tries to
# auto-install missing onnx/onnxruntime/onnxslim itself but that fails on
# externally-managed environments (PEP 668), so we check them all upfront.
PIP_PACKAGES="ultralytics onnx onnxruntime onnxslim"
MISSING=""
for mod in ultralytics onnx onnxruntime onnxslim; do
  python3 -c "import $mod" 2>/dev/null || MISSING="$MISSING $mod"
done

if [ -n "$MISSING" ]; then
  echo "WARNING: missing Python modules:$MISSING"
  if [ -t 0 ]; then
    read -r -p "Install them now with 'pip install --user --break-system-packages $PIP_PACKAGES'? [y/N] " reply
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      pip install --user --break-system-packages $PIP_PACKAGES
      MISSING=""
      for mod in ultralytics onnx onnxruntime onnxslim; do
        python3 -c "import $mod" 2>/dev/null || MISSING="$MISSING $mod"
      done
    fi
  fi
fi

if [ -n "$MISSING" ]; then
  echo "Skipping YOLO11 export - still missing:$MISSING"
  echo "To install manually: pip install --user --break-system-packages $PIP_PACKAGES"
  exit 0
fi

echo "=== Exporting YOLO11n to ONNX ==="
# A failed export must not stop the build (set -e would): warn, remove any
# half-written file and let the example report the missing model.
if ! python3 export_model.py; then
  echo "WARNING: YOLO11 export failed; the example will report the missing model."
  rm -f "$ONNX_FILE" "$NAMES_FILE"
  exit 0
fi

echo ""
echo "=== Done! ==="
echo "Files in $MODEL_DIR/:"
ls -lh "$MODEL_DIR/"

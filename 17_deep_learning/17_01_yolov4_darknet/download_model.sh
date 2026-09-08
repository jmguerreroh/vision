#!/bin/bash
# Download YOLOv4-tiny model files for OpenCV DNN
# Author: José Miguel Guerrero Hernández

set -e

CFG_DIR="../../data/models/yolov4"
mkdir -p "$CFG_DIR"

echo "=== Downloading YOLOv4-tiny model ==="

# Download config file
if [ ! -f "$CFG_DIR/yolov4-tiny.cfg" ]; then
  echo "Downloading yolov4-tiny.cfg..."
  wget -q --show-progress -O "$CFG_DIR/yolov4-tiny.cfg" \
    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
else
  echo "yolov4-tiny.cfg already exists, skipping."
fi

# Download weights file (~24 MB)
if [ ! -f "$CFG_DIR/yolov4-tiny.weights" ]; then
  echo "Downloading yolov4-tiny.weights (~24 MB)..."
  wget -q --show-progress -O "$CFG_DIR/yolov4-tiny.weights" \
    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
else
  echo "yolov4-tiny.weights already exists, skipping."
fi

# Download COCO class names
if [ ! -f "$CFG_DIR/coco.names" ]; then
  echo "Downloading coco.names..."
  wget -q --show-progress -O "$CFG_DIR/coco.names" \
    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
else
  echo "coco.names already exists, skipping."
fi

echo ""
echo "=== Done! ==="
echo "Files in $CFG_DIR/:"
ls -lh "$CFG_DIR/"

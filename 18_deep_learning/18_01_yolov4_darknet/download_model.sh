#!/bin/bash
# Download YOLOv4-tiny model files for OpenCV DNN
# Author: José Miguel Guerrero Hernández
#
# This script runs as a PRE_BUILD step, so it must never break the build: with
# no network it warns and exits 0, like the download scripts of 18_02 and 18_03.
#
# It also refuses to leave a half-written file behind. `wget -O` creates the
# destination before it knows whether the transfer will work, so a failed
# download used to leave a 0-byte file that the next run reported as "already
# exists, skipping". That file was never repaired, not even once the network
# came back.

set -e

CFG_DIR="../../data/models/yolov4"
mkdir -p "$CFG_DIR"

FALTAN=""

# wget if it is there, curl otherwise: a fresh Ubuntu ships neither, and saying
# only "could not download" hid the real reason, which was the missing tool.
if command -v wget > /dev/null 2>&1; then
  transferir() { wget -q --show-progress -O "$1" "$2"; }
elif command -v curl > /dev/null 2>&1; then
  transferir() { curl -fL --progress-bar -o "$1" "$2"; }
else
  echo "WARNING: neither wget nor curl is installed, so the YOLOv4-tiny model"
  echo "cannot be downloaded. Install one (sudo apt install wget) and re-run"
  echo "'bash download_model.sh' in this folder. The build continues."
  exit 0
fi

# Fetch a file only if it is not already there AND complete. Returns non-zero on
# failure, after removing whatever the download left, so the next run retries.
descargar() {
  local destino="$1" url="$2" tamano_minimo="$3" descripcion="$4"

  if [ -s "$destino" ]; then
    local bytes
    bytes=$(stat -c%s "$destino")
    if [ "$bytes" -ge "$tamano_minimo" ]; then
      echo "$(basename "$destino") already exists, skipping."
      return 0
    fi
    echo "$(basename "$destino") is only $bytes bytes, too small: downloading again."
    rm -f "$destino"
  fi

  echo "Downloading $(basename "$destino") ($descripcion)..."
  if transferir "$destino" "$url"; then
    local bytes
    bytes=$(stat -c%s "$destino")
    if [ "$bytes" -ge "$tamano_minimo" ]; then
      return 0
    fi
    echo "WARNING: $(basename "$destino") came out as $bytes bytes, expected at least $tamano_minimo."
  else
    echo "WARNING: could not download $(basename "$destino")."
  fi
  rm -f "$destino"          # nunca dejar un fichero a medias
  return 1
}

echo "=== Downloading YOLOv4-tiny model ==="

BASE_RAW="https://raw.githubusercontent.com/AlexeyAB/darknet/master"
BASE_REL="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre"

descargar "$CFG_DIR/yolov4-tiny.cfg"     "$BASE_RAW/cfg/yolov4-tiny.cfg"  1000     "~3 KB"  || FALTAN="$FALTAN yolov4-tiny.cfg"
descargar "$CFG_DIR/yolov4-tiny.weights" "$BASE_REL/yolov4-tiny.weights"  20000000 "~24 MB" || FALTAN="$FALTAN yolov4-tiny.weights"
descargar "$CFG_DIR/coco.names"          "$BASE_RAW/data/coco.names"      500      "~1 KB"  || FALTAN="$FALTAN coco.names"

if [ -n "$FALTAN" ]; then
  echo ""
  echo "Skipping the YOLOv4-tiny download - still missing:$FALTAN"
  echo "The build continues; 18_01_yolov4_darknet will report the missing model"
  echo "when you run it. Re-run 'bash download_model.sh' in this folder once you"
  echo "have network access."
  exit 0
fi

echo ""
echo "=== Done! ==="
echo "Files in $CFG_DIR/:"
ls -lh "$CFG_DIR/"

# A fresh Ubuntu with exactly what the README installs, plus what a headless
# run needs. The repository is not copied in: run_examples.py takes it from a
# mount, so the same image serves every commit.
#
#   docker build -f tools/ci/Dockerfile --build-arg UBUNTU=26.04 -t vision-ci:26.04 tools/ci

ARG UBUNTU=24.04
FROM ubuntu:${UBUNTU} AS deps

ENV DEBIAN_FRONTEND=noninteractive

# README, "From packages": build tools, OpenCV and PCL. wget for the model of
# 18_01 and pip for the exports of 18_02 and 18_03.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config \
      libopencv-dev libpcl-dev \
      wget ca-certificates python3-pip \
    # Not in the README, only for the CI: a virtual display, software OpenGL
    # for the VTK windows of PCL and viz, and the compiler of the autopilot.
      xvfb xauth libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# README, "Deep learning models". torch comes from the CPU index first: the
# default wheel drags several GB of CUDA that a CI runner cannot use, and
# ultralytics then finds its torch already installed.
ARG WITH_DL=1
RUN if [ "$WITH_DL" = 1 ]; then \
      pip install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
      pip install --no-cache-dir --break-system-packages \
        ultralytics onnx onnxruntime onnxslim onnxscript; \
    fi

FROM deps

# The autopilot is compiled here, against the OpenCV and PCL of this image: once
# for the OpenCV examples and once, linked to PCL and VTK, for the PCL ones.
COPY autopilot.cpp /opt/autopilot/
RUN cd /opt/autopilot && \
    CXX="g++ -std=c++17 -O2 -Wall -Wextra -shared -fPIC" && \
    LIB=/usr/lib/x86_64-linux-gnu && \
    $CXX -o libautopilot.so autopilot.cpp $(pkg-config --cflags opencv4) -ldl && \
    $CXX -DAUTOPILOT_PCL -o libautopilot_pcl.so autopilot.cpp \
      $(pkg-config --cflags opencv4) -I/usr/include/eigen3 \
      -I$(ls -d /usr/include/pcl-* | head -1) -I$(ls -d /usr/include/vtk-* | head -1) \
      -lpcl_visualization -lpcl_common \
      $(ls $LIB/libvtkCommonCore-*.so $LIB/libvtkRenderingCore-*.so) -ldl

RUN git config --global --add safe.directory '*'

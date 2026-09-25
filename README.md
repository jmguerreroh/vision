# Computer Vision Examples

[![CI](https://github.com/jmguerreroh/vision/actions/workflows/ci.yml/badge.svg)](https://github.com/jmguerreroh/vision/actions/workflows/ci.yml)

Code examples for the Computer Vision subject of the Robotics Software Engineering Degree at URJC, using C++, OpenCV, and the Point Cloud Library (PCL).

---

## Quick start

On Ubuntu, with the distribution packages:

```bash
sudo apt update
sudo apt install build-essential cmake git pkg-config libopencv-dev libpcl-dev
git clone https://github.com/jmguerreroh/vision.git && cd vision
echo "export OPENCV_SAMPLES_DATA_PATH=$(pwd)/data/" >> ~/.bashrc && source ~/.bashrc
cmake -B vision_examples/build && cmake --build vision_examples/build
cd vision_examples/bin && ./03_01_read_image
```

The rest of this README explains each step, the optional pieces (chapter 18
models, ROS 2) and how to build from source.

---

## Repository structure

The examples are organised by chapter and follow the order in which the book
introduces the material. There are 80 in total: 75 numbered `NN_MM` examples,
where `NN` is the book chapter, plus the 5 ROS 2 packages of chapter 19, which
are named after the package instead of numbered because `colcon` builds them by
name. The folder column below is the authoritative mapping between a book
chapter and its code.

| Chapter | Folder | Topic | Examples |
|---------|--------|-------|----------|
| 02 | `02_image_formation` | Image formation | thin lens and depth of field |
| 03 | `03_digital_image_and_color` | The digital image and color | read image, color spaces, Mat copy & ROI, pixel access, video capture |
| 04 | `04_pixel_and_filtering` | Pixel operations and spatial filtering | point ops, convolution, bitwise, intensity transforms, smoothing |
| 05 | `05_histogram` | The histogram | histogram equalization, matching, comparison |
| 06 | `06_frequency` | Frequency-domain transforms | DFT, DCT, wavelet denoising, Gabor bank, homomorphic filter |
| 07 | `07_geometric_and_registration` | Geometric transforms and registration | affine transforms, perspective correction |
| 08 | `08_edge_detection` | Edge detection | Sobel, Canny, Laplacian, contour extraction, chain code |
| 09 | `09_model_fitting` | Model fitting | Hough lines, Hough circles |
| 10 | `10_region_segmentation` | Region segmentation | threshold, connected components, color segmentation |
| 11 | `11_morphological_operations` | Morphological operations | erode/dilate, opening/closing, gradient, hit-or-miss, skeletonization, flood fill, top-hat illumination, distance + watershed |
| 12 | `12_region_descriptors` | Region descriptors | region moments, Hu moments, convex hull |
| 13 | `13_keypoints` | Keypoints | Harris, Shi-Tomasi, ORB, RANSAC matching |
| 14 | `14_camera_calibration` | Camera geometry and calibration | chessboard calibration, pose estimation (PnP), stereo calibration + rectification |
| 15 | `15_3d_and_point_clouds` | 3D vision and point clouds | epipolar geometry, disparity, disparity to point cloud, OpenCV ICP, PCL I/O, visualizers, PCL ICP, RANSAC model fitting, registration, correspondence, plane + clustering |
| 16 | `16_optical_flow_and_tracking` | Optical flow and tracking | frame difference, Lucas-Kanade, Farneback dense flow, background subtraction, Kalman tracking, object tracking |
| 17 | `17_classical_ml` | Classical machine learning | k-NN, SVM, digit classification, k-means, classifier comparison, self-organizing map |
| 18 | `18_deep_learning` | Deep learning | YOLOv4, YOLO11, semantic segmentation |
| 19 | `19_vision_ros2` | Vision in ROS 2 | opencv_demo (cv_bridge), transport_demo (image_transport), sync_demo (message_filters), pcl_demo (pcl_conversions), launch_demo (built with `colcon`, see [ROS 2 examples](#ros-2-examples-chapter-19)) |

Every example is **self-contained and runnable on its own**: they can be run in
any order and none of them needs another to have run first. Two of them are
linked on purpose, and neither link is required:

- `15_03_stereo_to_pointcloud` accepts `--calib=stereo_calibration.yml`, the
  file that `14_03_stereo_calibration` writes. With it the pair is rectified
  and the cloud comes out in real units; without it the example falls back to
  an assumed rig and says so.
- `15_05_pcl_write` writes the `test_pcd.pcd` that `15_06_pcl_read` reads. That
  file is kept under version control, so `15_06` also works on a fresh clone.
  The generator of `15_05` is seeded, so running it rewrites the file byte for
  byte instead of producing a spurious change.

What does follow the book order is the material each one assumes you have
already read, which is the reason for studying them from beginning to end.

---

## Requirements

The examples use modern features and require a compiler that supports **C++17**
or higher.

The build requires **OpenCV 4**. **PCL** is optional: the top-level
`CMakeLists.txt` looks for PCL 1.10 and, if it is not there, warns and skips the
nine PCL examples of chapter 15 instead of failing. The other 69 targets build
without it.

The examples were developed on **Ubuntu 24.04** (OpenCV 4.6.0, PCL 1.14.0) and
are also tested on **Ubuntu 26.04** (OpenCV 4.10.0, PCL 1.15.1), both from a
fresh install: all of them build on both. One runs only on the second:
`18_02`, whose YOLO11 model in ONNX needs **OpenCV 4.9 or newer**, because
earlier ONNX readers do not understand the `Split` node the way YOLO11 writes
it. With an older OpenCV the example reports exactly that and exits, and
`18_01` covers the same ground with a model that loads anywhere.

Some examples require the **opencv_contrib** modules (`ximgproc`, `aruco`,
`surface_matching`, `viz`, `tracking`). If you installed OpenCV from the
distribution package these are usually included; if you built OpenCV from
source, follow [OpenCV from source](#opencv-from-source) and pass
`OPENCV_EXTRA_MODULES_PATH`. The examples that need them are:
`11_05_skeletonization`, `14_02_pose_estimation`, `15_02_stereo_disparity`,
`15_04_opencv_icp` and `16_06_object_tracking`.

The chapter 19 examples also need ROS 2 (Jazzy or Lyrical); see
[ROS 2 examples](#ros-2-examples-chapter-19).

---

## Installation

### From packages (recommended)

**Build tools** (a fresh Ubuntu does not ship them):

```bash
sudo apt update
sudo apt install build-essential cmake git pkg-config
```

**OpenCV:**

```bash
sudo apt update
sudo apt install libopencv-dev
```

Verify:

```bash
pkg-config --modversion opencv4
```

**PCL:**

```bash
sudo apt update
sudo apt install libpcl-dev
```

Verify:

```bash
dpkg -s libpcl-dev | grep Version
```

> If you have ROS installed, OpenCV and PCL are likely already available through your ROS distribution.

**Deep learning models (chapter 18, optional):**

The three examples of chapter 18 get their model on first build, each with the
`download_model.sh` that lives beside its source. None of them can fail the
build: if the model cannot be obtained they warn and let the build finish, and
the example reports the missing model when you run it.

`18_01_yolov4_darknet` **downloads about 24 MB** from
`github.com/AlexeyAB/darknet`, so the first build needs network access and
`wget` or `curl` (`sudo apt install wget`; a fresh Ubuntu has neither). A failed
download leaves nothing behind, and re-running the script retries it.

`18_02_yolo_ultralytics` and `18_03_semantic_segmentation` export their ONNX
model locally instead, with a `download_model.sh`/`export_model.py` pair. If the
required Python packages aren't installed, the script prints a warning and
skips the export.

`pip` itself is not installed by default either:

```bash
sudo apt install python3-pip
pip install --user --break-system-packages ultralytics onnx onnxruntime onnxslim  # 18_02
pip install --user --break-system-packages torch torchvision onnxscript           # 18_03
```

The 18_03 export has been tested with torch 2.3.1 and 2.14. Since torch 2.5,
`torch.onnx.export` accepts a `dynamo` argument, and since 2.9 the dynamo exporter
is the default; whenever the argument exists the script asks for the legacy
exporter, because the dynamo one produces a graph that OpenCV's ONNX importer
cannot read. If an export fails for any other reason,
the script warns and the build still finishes.

Re-run `bash download_model.sh` inside each example's folder (or rebuild its
target) afterwards to generate the missing model files.

<details>
<summary><strong>Installation from source</strong></summary>

### OpenCV from source

1. Install dependencies:

```bash
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb-dev libdc1394-dev
```

2. Clone OpenCV and the contrib modules:

```bash
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

3. Configure:

```bash
cd ~/opencv_build/opencv
mkdir build && cd build
```

Without CUDA:

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    ..
```

With CUDA (adjust `CUDA_ARCH_BIN` for your GPU — see https://developer.nvidia.com/cuda-gpus):

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=8.6 \
    -D WITH_CUBLAS=1 \
    ..
```

4. Compile and install:

```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

5. Add to `~/.bashrc` and reload:

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/' >> ~/.bashrc
source ~/.bashrc
```

---

### PCL from source

See the [official guide](https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html) for details.

1. Download the [latest stable release](https://github.com/PointCloudLibrary/pcl/releases) and extract it, or clone the repository:

```bash
mkdir ~/pcl_build && cd ~/pcl_build
git clone --recursive https://github.com/PointCloudLibrary/pcl.git
```

2. Configure, compile, and install:

```bash
cd ~/pcl_build/pcl
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

</details>

### Data path

Whichever way you installed the libraries, set the environment variable so the
examples can find the data folder. Run this from the repository root:

```bash
echo "export OPENCV_SAMPLES_DATA_PATH=$(pwd)/data/" >> ~/.bashrc
source ~/.bashrc
```

---

## Building

### All examples at once (recommended)

A top-level `CMakeLists.txt` compiles every example in one step and places all executables in the `vision_examples/bin/` folder, named after their source directory:

```bash
cmake -B vision_examples/build
cmake --build vision_examples/build
```

Executables are in `vision_examples/bin/`. Run them from that folder:

```bash
cd vision_examples/bin
./03_01_read_image
./06_01_dft_frequencies
./15_02_stereo_disparity
```

> Note: the default inputs are written as `../../data/...`, and
> `cv::samples::findFile` looks for them from the current working directory.
> That path resolves to the `data/` folder of the repository from
> `vision_examples/bin/` and from the folder of the example itself, but **not
> from the repository root**. Either way of building works without touching the
> paths, as long as the example is launched from one of those two folders.

### A single example (OpenCV)

Each OpenCV example also has its own `Makefile`. The executable takes the name
of its folder, exactly like the one the top-level build produces, so both ways
of compiling give the same binary:

```bash
cd 08_edge_detection/08_02_canny_edges
make
./08_02_canny_edges
```

### A single example (PCL)

Each PCL example has its own `CMakeLists.txt`:

```bash
cd 15_3d_and_point_clouds/15_09_pcl_icp
cmake -B build
cmake --build build
./build/15_09_pcl_icp
```

---

## Running the examples

### Command-line interface

Every example follows the same convention, so any of them can be run without
reading its source first. From `vision_examples/bin/`:

```bash
./example                 # runs with its default input, taken from data/
./example my_image.jpg    # overrides the input
./example --help          # prints what the example accepts and its defaults
```

OpenCV examples use `cv::CommandLineParser`; PCL examples use PCL's own
`pcl::console` parser. Both accept `-h` and `--help`. Inputs are **positional
and optional**: an example with no arguments works, with three exceptions. `15_08_pcl_advanced_visualizer` is a menu of seven demos and prints
that menu when called with no option; pick one, for instance `-s`. And
`03_05_video_capture` and `06_03_wavelet_denoising` are real-time examples that
open the default camera when called with no argument; without a camera, give
them a file, for instance `../../data/vtest.avi` and
`../../data/starry_night.png`.

The few examples that write a file take the destination on the command line and
default to the current directory: `--out` in `14_01`, `14_03`, `15_03` and
`17_02`, and `--dst_path` (plus `--dst_raw_path` and `--dst_conf_path`) in
`15_02`. Note that `15_03_stereo_to_pointcloud` writes a `cloud.ply` of about
40 MB. Two examples write without asking, and neither takes a destination:
`03_05_video_capture` always saves what it captures to `output.avi` in the
current directory, and `15_11_pcl_registration` writes one
`result_00N.pcd` per registered pair into `data/pcl_data/`, next to the
captures it read. Both names are fixed, and the `result_*.pcd` are ignored by
`.gitignore`.

### Default images

The examples default to the **same photographs the book uses in its figures**,
so running one without arguments reproduces what the reader has just seen
printed. `data/building_facade.png`, `coins.png`, `chess.png`, `smarties.png`,
`aerial_view.png`, `starry_night.png` and `futbol.png` are the very files that
the figure-generating scripts of the book read.

The optical flow examples of chapter 16 default to the same video, the overhead
shot of a busy square that the book credits to Pexels 853889.

Those photographs are around 1400 px on the long side, and the video is Full HD,
which does not fit on a normal screen once an example opens four or five
windows. Every example that uses them reduces **only what it sends to the
screen**, with `INTER_AREA` and a long side of 800 px; the processing always
runs at full resolution. The reduction is a no-op on smaller inputs, so passing
your own image changes nothing.

Text is written on the reduced copy, or with the font raised by the same factor
when it is a label anchored to a region, so that it stays readable instead of
shrinking with the picture.

The one exception to processing at full resolution is `16_03_dense_flow`:
Farneback costs 392 ms per frame at 1920x1080, ten times the 40 ms a 25 fps
video allows, so it reduces the frames by `--scale` (0.5 by default, the same
factor the book uses for its figures) before computing the flow. Pass
`--scale=1.0` to see the difference.

---

## ROS 2 examples (Chapter 19)

The Chapter 19 examples live here like every other chapter, under
`19_vision_ros2/`, but they are not built with the rest. They are ROS 2
packages, not standalone programs: they need a workspace, they are built with
`colcon` and they are run with `ros2 run`. The top-level `CMakeLists.txt`
ignores that folder on purpose, so the repository still builds for anyone who
only wants the OpenCV examples and has no ROS 2 installed.

There are five packages: `opencv_demo`, `transport_demo`, `sync_demo` and
`pcl_demo`, one per piece developed in the chapter, plus `launch_demo`, which
holds the launch file that chains the `depth_image_proc` nodes to produce the
cloud `pcl_demo` consumes.

```bash
cd <repository root>
sudo rosdep init && rosdep update      # only once per machine, if never done
rosdep install --from-paths 19_vision_ros2 --ignore-src -r -y
colcon build --base-paths 19_vision_ros2 --symlink-install
source install/setup.bash
```

`--base-paths` is what keeps `colcon` from descending into the rest of the
repository, where it would find the OpenCV/PCL project of the other chapters.

The packages are tested with **ROS 2 Jazzy** (Ubuntu 24.04) and **ROS 2 Lyrical**
(Ubuntu 26.04). Where the ROS API changed between them (the removal of
`ament_target_dependencies`, the `.hpp` headers of `message_filters`, and QoS
passed as `rclcpp::QoS`), the code builds on both without warnings.

> **Large messages on Lyrical.** With its default transport, the Fast DDS of
> Lyrical often drops large best-effort messages: a 640×480 point cloud (about
> 5 MB) can take tens of seconds to get through to `pcl_demo`, or not arrive at
> all, while a small cloud arrives at once. Jazzy does not show this. Enabling
> the large-data mode of Fast DDS, in every terminal that runs a node, fixes it:
>
> ```bash
> export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
> ```
>
> Measured over five runs each: without it, 4 of 5 clouds arrived, after 7 to
> 55 s; with it, 5 of 5 in 1 to 2 s.

Requirements, beyond one of those distributions: `cv_bridge`,
`image_transport` (plus `image-transport-plugins`), `message_filters`,
`pcl_ros` and `depth_image_proc`. `rosdep` installs them from the manifests, or
by hand:

```bash
sudo apt install ros-${ROS_DISTRO}-cv-bridge \
                 ros-${ROS_DISTRO}-image-transport \
                 ros-${ROS_DISTRO}-image-transport-plugins \
                 ros-${ROS_DISTRO}-message-filters \
                 ros-${ROS_DISTRO}-pcl-ros \
                 ros-${ROS_DISTRO}-depth-image-proc
```

| Package | Executable | Subscribes to | Publishes | What it shows |
|---|---|---|---|---|
| `opencv_demo` | `opencv_processing` | `/color/image` | `/image_processed` | The `cv_bridge` round trip: ROS message to `cv::Mat` and back, keeping the original header |
| `transport_demo` | `transport_processing` | `/color/image` | `image_processed` (+ transport sub-topics) | The same node through `image_transport`: one publisher, several wire formats |
| `sync_demo` | `sync_processing` | `/left/image`, `/right/image` | (displays) | `message_filters` with an `ApproximateTime` policy: one callback, two images already paired |
| `pcl_demo` | `pcl_processing` | `/stereo/points` | `/pcl_processed` | `pcl_conversions`: `PointCloud2` to `pcl::PointCloud` and back. The gap between both conversions is where your PCL algorithm goes |
| `launch_demo` | (launch only) | | `/stereo/points` | Chains the `depth_image_proc` nodes that produce the cloud `pcl_demo` consumes |

Every node works the same against a live camera or against a recording:

```bash
ros2 bag record /color/image /color/camera_info /stereo/depth -o session
ros2 bag play session
```

Things worth trying:

- **`opencv_demo`**: ask `toCvCopy` for `BGR8` on a depth topic and watch the
  `cv_bridge` exception; then ask for `BGR8` on an `rgb8` camera and notice that
  nothing breaks, because `cv_bridge` converts.
- **`transport_demo`**: compare `ros2 topic bw /image_processed` with
  `ros2 topic bw /image_processed/compressed`. The saving depends on the scene,
  not on the format in the abstract.
- **`sync_demo`**: drop the queue size to 1 and count how many pairs are lost;
  switch the policy to `ExactTime` and watch the callback stop firing unless the
  cameras share a hardware trigger.
- **`pcl_demo`**: drop a `VoxelGrid` filter between the two conversions and
  compare `ros2 topic hz` on input and output.

A note on QoS: these nodes use plain `rclcpp::SensorDataQoS()`, which is *best
effort*, on both ends. Forcing it to `.reliable()` on the subscriber makes it
incompatible with any publisher that offers best effort, which is what most
camera drivers do, and there is no error message when that happens: the topic is
listed, `ros2 topic hz` reports data, and the callback simply never runs. Check
the actual profiles with:

```bash
ros2 topic info /color/image --verbose
```

---

## FAQ

**`fatal error: opencv2/opencv.hpp: No such file or directory`**

OpenCV headers are not found. Make sure `libopencv-dev` is installed. If installed from source, add the pkg-config path:

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

---

**`error while loading shared libraries: libopencv_core.so.X`**

The dynamic linker cannot find the OpenCV libraries. If you installed from source to a custom path, make sure it is in `LD_LIBRARY_PATH`. For standard source installations (`/usr/local`), simply update the linker cache:

```bash
sudo ldconfig
```

---

**`pkg-config: command not found`**

```bash
sudo apt install pkg-config
```

---

**`VideoCapture` does not open the camera**

1. Try different indices (`0`, `1`, ...).
2. Add your user to the `video` group and log out/in:
   ```bash
   sudo usermod -aG video $USER
   ```
3. Make sure the `v4l2` module is loaded:
   ```bash
   sudo modprobe v4l2
   ```

---

**PCL visualizer window does not open or crashes**

This is usually a VTK/OpenGL issue:

1. Install or update VTK: `sudo apt install libvtk9-dev`
2. On virtual machines or remote sessions, force software rendering:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   ```
3. Compiling PCL from source (instead of packages) often resolves VTK 9.x compatibility issues.

---

**CMake cannot find PCL (`Could not find PCL`)**

Make sure `libpcl-dev` is installed. If installed from source, hint CMake to the correct path from inside your `build` folder:

```bash
mkdir build && cd build
cmake -DPCL_DIR=/usr/local/share/pcl-<version> ..
```

---

**CUDA not detected when building OpenCV from source**

Make sure `nvcc` is in your `PATH` and the CUDA libraries are accessible:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Also verify `CUDA_ARCH_BIN` matches your GPU's compute capability at https://developer.nvidia.com/cuda-gpus.

---

## Checking the repository

`tools/check_repo.py` verifies the things that drift when a chapter is renamed
or an example moves: that every example on disk is built by the top-level
`CMakeLists.txt`, that each one produces a binary named after its folder
whichever way it is compiled, that no header cites an executable or an example
that does not exist, that the default input paths point at files that are
really there, and that every example answers `-h` and `--help`.

```bash
python3 tools/check_repo.py
```

It exits non-zero on the first inconsistency, so it can be used in CI. Run it
after building: the model files of chapter 18 it checks for are generated by the
build.

### Continuous integration

On every push, `.github/workflows/ci.yml` builds everything on fresh Ubuntu
24.04 and 26.04 images and the ROS 2 packages on Jazzy and Lyrical. Once a week
it also runs every example, unattended. The same checks run locally, in the same
Docker images:

```bash
tools/ci/run_local.sh              # 24.04, 26.04, jazzy and lyrical
tools/ci/run_local.sh 26.04        # only one of them
```

---

## About

This project was made by [Jose Miguel Guerrero], Associate Professor at [Universidad Rey Juan Carlos].

Copyright &copy; 2020-2026.

[![Twitter](https://img.shields.io/badge/follow-@jm__guerrero-green.svg)](https://twitter.com/jm__guerrero)

## License

This work is licensed under the terms of the [MIT license](https://opensource.org/license/mit).

[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Universidad Rey Juan Carlos]: https://www.urjc.es/
[Jose Miguel Guerrero]: https://sites.google.com/view/jmguerrero

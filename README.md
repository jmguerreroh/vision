# Computer Vision Examples

Code examples for the Computer Vision subject of the Robotics Software Engineering Degree at URJC, using C++, OpenCV, and the Point Cloud Library (PCL).

---

## Usage

First, set the environment variable so examples can find the data folder. Run this from the repository root:

```bash
echo "export OPENCV_SAMPLES_DATA_PATH=$(pwd)/data/" >> ~/.bashrc
source ~/.bashrc
```

> Note: The examples use modern features and require a compiler that supports **C++17** or higher.

> Note: Developed against OpenCV 4.6.0 and PCL 1.14.0. The build requires
> OpenCV 4 and at least PCL 1.10, which is what the top-level `CMakeLists.txt`
> asks for.

> Note: Some examples require the **opencv_contrib** modules (`ximgproc`,
> `aruco`, `surface_matching`, `viz`). If you installed OpenCV from the
> distribution package these are usually included; if you built OpenCV from
> source, follow the *Installation from source* section below and pass
> `OPENCV_EXTRA_MODULES_PATH`. The examples that need them are:
> `10_05_skeletonization`, `13_02_pose_estimation`, `14_02_stereo_disparity`
> and `14_04_opencv_icp`.

### Building all examples at once (recommended)

A top-level `CMakeLists.txt` compiles every example in one step and places all executables in the `vision_examples/bin/` folder, named after their source directory:

```bash
cmake -B vision_examples/build
cmake --build vision_examples/build
```

Executables are in `vision_examples/bin/`. For example:

```bash
./02_01_read_image
./05_01_dft_frequencies
./14_02_stereo_disparity
```

> Note: the default inputs are written as `../../data/...`, which resolves to
> the `data/` folder of the repository both from `vision_examples/bin/` and
> from the folder of the example itself. Either way of building works without
> touching the paths.

### Command-line interface

Every example follows the same convention, so any of them can be run without
reading its source first:

```bash
./example                 # runs with its default input, taken from data/
./example my_image.jpg    # overrides the input
./example --help          # prints what the example accepts and its defaults
```

OpenCV examples use `cv::CommandLineParser`; PCL examples use PCL's own
`pcl::console` parser. Both accept `-h` and `--help`. Inputs are **positional
and optional**: an example with no arguments always works. The few examples
that write a file take the destination on the command line and default to the
current directory: `--out` in `13_01`, `13_03`, `14_03` and `16_02`, and
`--dst_path` (plus `--dst_raw_path` and `--dst_conf_path`) in `14_02`.

### Building a single example (OpenCV)

Each OpenCV example also has its own `Makefile`. The executable takes the name
of its folder, exactly like the one the top-level build produces, so both ways
of compiling give the same binary:

```bash
cd 07_edge_detection/07_02_canny_edges
make
./07_02_canny_edges
```

### Building a single example (PCL)

Each PCL example has its own `CMakeLists.txt`:

```bash
cd 14_3d_and_point_clouds/14_09_pcl_icp
cmake -B build
cmake --build build
./build/14_09_pcl_icp
```

### Building the ROS 2 examples (Chapter 18)

The canonical home of the Chapter 18 examples is a separate repository,
<https://github.com/jmguerreroh/vision_ros2>, which is the one the book points
to. They are ROS 2 packages, not standalone programs: they need a workspace,
they are built with `colcon` and they are run with `ros2 run`, so mixing them
into this build would force a ROS 2 installation on everyone who just wants to
compile the OpenCV examples.

A copy is mirrored here under `18_vision_ros2/` for convenience, and the
top-level `CMakeLists.txt` ignores it on purpose, so the rest of the repository
still builds without ROS 2. If the two ever disagree, `vision_ros2` is the one
to trust.

```bash
cd <repository root>
rosdep install --from-paths 18_vision_ros2 --ignore-src -r -y
colcon build --base-paths 18_vision_ros2 --symlink-install
source install/setup.bash
```

`--base-paths` is what keeps `colcon` from descending into the rest of the
repository, where it would find the OpenCV/PCL project of the other chapters.

Requirements, beyond a current ROS 2 distribution: `cv_bridge`,
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

### Default images

The examples default to the **same photographs the book uses in its figures**,
so running one without arguments reproduces what the reader has just seen
printed. `data/building_facade.png`, `coins.png`, `chess.png`, `smarties.png`,
`aerial_view.png`, `starry_night.png` and `futbol.png` are the very files that
the figure-generating scripts of the book read.

The optical flow examples of chapter 15 default to the same video, the overhead
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

The one exception to processing at full resolution is `15_03_dense_flow`:
Farneback costs 392 ms per frame at 1920x1080, ten times the 40 ms a 25 fps
video allows, so it reduces the frames by `--scale` (0.5 by default, the same
factor the book uses for its figures) before computing the flow. Pass
`--scale=1.0` to see the difference.

### Checking the repository

`tools/check_repo.py` verifies the things that drift when a chapter is renamed
or an example moves: that every example on disk is built by the top-level
`CMakeLists.txt`, that each one produces a binary named after its folder
whichever way it is compiled, that no header cites an executable or an example
that does not exist, that the default input paths point at files that are
really there, and that every example answers `-h` and `--help`.

```bash
python3 tools/check_repo.py
```

It exits non-zero on the first inconsistency, so it can be used in CI.

---

## Repository structure

The examples are organised by chapter and follow the order in which the book
introduces the material. There are 80 in total: 75 numbered `NN_MM` examples,
where `NN` is the book chapter, plus the 5 ROS 2 packages of chapter 18, which
are named after the package instead of numbered because `colcon` builds them by
name. The folder column below is the authoritative mapping between a book
chapter and its code.

| Chapter | Folder | Topic | Examples |
|---------|--------|-------|----------|
| 02 | `02_image_formation` | Image formation | read image, color spaces, Mat copy & ROI, pixel access, video capture |
| 03 | `03_pixel_and_filtering` | Pixel operations and spatial filtering | point ops, convolution, bitwise, intensity transforms, smoothing |
| 04 | `04_histogram` | The histogram | histogram equalization, matching, comparison |
| 05 | `05_frequency` | Frequency-domain transforms | DFT, DCT, wavelet denoising, Gabor bank, homomorphic filter |
| 06 | `06_geometric_and_registration` | Geometric transforms and registration | affine transforms, perspective correction |
| 07 | `07_edge_detection` | Edge detection | Sobel, Canny, Laplacian, contour extraction, chain code |
| 08 | `08_model_fitting` | Model fitting | Hough lines, Hough circles |
| 09 | `09_region_segmentation` | Region segmentation | threshold, connected components, color segmentation |
| 10 | `10_morphological_operations` | Morphological operations | erode/dilate, opening/closing, gradient, hit-or-miss, skeletonization, flood fill, top-hat illumination, distance + watershed |
| 11 | `11_region_descriptors` | Region descriptors | region moments, Hu moments |
| 12 | `12_keypoints` | Keypoints | Harris, Shi-Tomasi, ORB, RANSAC matching |
| 13 | `13_camera_calibration` | Camera geometry and calibration | chessboard calibration, pose estimation (PnP), stereo calibration + rectification |
| 14 | `14_3d_and_point_clouds` | 3D vision and point clouds | epipolar geometry, disparity, disparity to point cloud, OpenCV ICP, PCL I/O, visualizers, PCL ICP, RANSAC model fitting, registration, correspondence, plane + clustering |
| 15 | `15_optical_flow_and_tracking` | Optical flow and tracking | frame difference, Lucas-Kanade, Farneback dense flow, background subtraction, Kalman tracking, object tracking |
| 16 | `16_classical_ml` | Classical machine learning | k-NN, SVM, digit classification, k-means, classifier comparison, self-organizing map |
| 17 | `17_deep_learning` | Deep learning | YOLOv4, YOLO11, semantic segmentation |
| 18 | `18_vision_ros2` | Vision in ROS 2 | opencv_demo (cv_bridge), transport_demo (image_transport), sync_demo (message_filters), pcl_demo (pcl_conversions), launch_demo (built with `colcon`, see above) |

Every example is **self-contained and runnable on its own**: they can be run in
any order and none of them needs another to have run first. Two of them are
linked on purpose, and neither link is required:

- `14_03_stereo_to_pointcloud` accepts `--calib=stereo_calibration.yml`, the
  file that `13_03_stereo_calibration` writes. With it the pair is rectified
  and the cloud comes out in real units; without it the example falls back to
  an assumed rig and says so.
- `14_05_pcl_write` writes the `test_pcd.pcd` that `14_06_pcl_read` reads. That
  file is kept under version control, so `14_06` also works on a fresh clone.
  The generator of `14_05` is seeded, so running it rewrites the file byte for
  byte instead of producing a spurious change.

What does follow the book order is the material each one assumes you have
already read, which is the reason for studying them from beginning to end.

---

## Installation

### From packages (recommended)

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

---

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

## About

This project was made by [Jose Miguel Guerrero], Associate Professor at [Universidad Rey Juan Carlos].

Copyright &copy; 2020-2026.

[![Twitter](https://img.shields.io/badge/follow-@jm__guerrero-green.svg)](https://twitter.com/jm__guerrero)

## License

This work is licensed under the terms of the [MIT license](https://opensource.org/license/mit).

[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Universidad Rey Juan Carlos]: https://www.urjc.es/
[Jose Miguel Guerrero]: https://sites.google.com/view/jmguerrero

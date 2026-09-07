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

> Note: This project uses OpenCV 4.6.0 and PCL 1.14.0.

> Note: Some examples require the **opencv_contrib** modules (`ximgproc`,
> `aruco`, `surface_matching`, `viz`). If you installed OpenCV from the
> distribution package these are usually included; if you built OpenCV from
> source, follow the *Installation from source* section below and pass
> `OPENCV_EXTRA_MODULES_PATH`. The examples that need them are:
> `09_05_skeletonization`, `10_02_pose_estimation`, `11_02_stereo_disparity`
> and `11_04_opencv_icp`.

### Building all examples at once (recommended)

A top-level `CMakeLists.txt` compiles every example in one step and places all executables in the `vision_examples/bin/` folder, named after their source directory:

```bash
cmake -B vision_examples/build
cmake --build vision_examples/build
```

Executables are in `vision_examples/bin/`. For example:

```bash
./02_01_read_image
./04_01_dft_frequencies
./11_02_stereo_disparity
```

> Note: default paths assume that the examples are running from the vision_examples/bin directory.

### Command-line interface

Every example follows the same convention, so any of them can be run without
reading its source first:

```bash
./example                 # runs with its default input, taken from data/
./example my_image.jpg    # overrides the input
./example --help          # prints what the example accepts and its defaults
```

OpenCV examples use `cv::CommandLineParser`; PCL examples use PCL's own
`pcl::console` parser and answer to `-h`. Inputs are **positional and
optional**: an example with no arguments always works.

### Building a single example (OpenCV)

Each OpenCV example also has its own `Makefile`:

```bash
cd example_folder
make
./executable
```

### Building a single example (PCL)

Each PCL example has its own `CMakeLists.txt`:

```bash
cd example_folder
cmake -B build
cmake --build build
./build/executable
```

### Building the ROS 2 examples (Chapter 12)

The examples of Chapter 12 live in `14_vision_ros2/` like every other chapter,
but they are **not** part of the build above: they are ROS 2 packages, not
standalone programs, so they are built with `colcon` and run with `ros2 run`.
The top-level `CMakeLists.txt` ignores them on purpose, so the rest of the
repository still builds without a ROS 2 installation.

```bash
cd <repository root>
rosdep install --from-paths 14_vision_ros2 --ignore-src -r -y
colcon build --base-paths 14_vision_ros2 --symlink-install
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

---

## Repository structure

The examples are organised by chapter and are meant to be studied **in order**: each one introduces a single main idea and only relies on concepts shown in earlier examples.

| Chapter | Topic | Examples |
|---------|-------|----------|
| 02 | Image formation | read image, Mat copy & ROI, pixel access, color spaces, video |
| 03 | Frequency-domain transforms | Fourier, DCT, wavelets, Gabor bank |
| 04 | Spatial, geometric and radiometric transforms | point ops, convolution, bitwise, affine transforms, perspective correction, smoothing, homomorphic filter, histogram equalization/matching/comparison |
| 05 | Edges and model fitting | Sobel, Canny, Laplacian, contours, chain code, Hough lines, Hough circles |
| 06 | Regions, descriptors and keypoints | threshold, connected components, color segmentation, moments, Hu, Harris, Shi-Tomasi, ORB, RANSAC matching |
| 07 | Morphology | erode/dilate, compound operations, gradient, hit-or-miss, skeleton, flood fill, top-hat illumination, distance + watershed |
| 08 | Camera calibration | chessboard calibration, ChArUco pose (PnP), **stereo calibration + rectification** |
| 09 | 3D vision and point clouds | **epipolar geometry**, disparity, disparity → point cloud, OpenCV ICP, PCL I/O, visualizers, ICP, RANSAC fitting, registration, correspondence, plane + clustering pipeline |
| 10 | Optical flow | frame difference, Lucas-Kanade, Farneback, background subtraction, Kalman tracking, **CamShift vs CSRT tracking** |
| 11 | Pattern recognition | KNN, SVM, digit classification + metrics, K-Means, ML comparison, **self-organizing map**, YOLOv4, YOLO11, **semantic segmentation** |
| 12 | Vision in ROS 2 | **cv_bridge node, image_transport, message_filters sync, PCL conversion, depth_image_proc launch** (built with `colcon`, see above) |

The order of the examples is the order in which the book introduces the
material, so the numbering can be followed from beginning to end. Examples in
**bold** were added to close gaps between the book text and the code. Whenever an example builds on a previous one, its header comment names
the earlier example it depends on.

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

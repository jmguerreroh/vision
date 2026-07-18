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

### Building all examples at once (recommended)

A top-level `CMakeLists.txt` compiles every example in one step and places all executables in the `vision_examples/bin/` folder, named after their source directory:

```bash
cmake -B vision_examples/build
cmake --build vision_examples/build
```

Executables are in `vision_examples/bin/`. For example:

```bash
./02_01_read_image
./03_01_fourier_frequencies
./08_01_stereo_disparity
```

> Note: default paths assume that the examples are running from the vision_examples/bin directory.

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

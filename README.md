# Computer Vision Examples

This project contains code examples created in Visual Studio Code for Computer Vision using C++ & OpenCV & Point Cloud Library (PCL). These examples are created for the Computer Vision Subject of Robotics Software Engineering Degree at URJC.

## Compiling examples and creating executables

Add an environment variable to find the samples into your `.bashrc` file
```
export OPENCV_SAMPLES_DATA_PATH=/<repo-folder>/data/
```

OpenCV examples use a Makefile, use `make` command to compile and create the executable.
```
cd example_folder
make
./executable
```

PCL examples use a CMakeList.txt, so you should create the Makefile using `cmake` and create the executable using `make`. Is a good practice to create it in a build folder.
```
cd example_folder
mkdir build && cd build && cmake .. && make
./executable
```

# Installation from source

If you have ROS packages installed, you may skip the following OpenCV and PCL installation steps, as OpenCV and PCL might already be included in your ROS distribution.

## OpenCV installation from source (do not install without supervision)

Probably OpenCV is installed, but it's a good practice to install it as follows.

Open an Ubuntu terminal and follow the next steps:

1. Install dependencies:
```
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-dev
```

2. Clone OpenCV and Contrib repositories:
```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

3. Create temporary build directory:
```
cd ~/opencv_build/opencv
mkdir build
cd build
```

4. Setup OpenCV:

Without CUDA:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..\
    -D OPENCV_ENABLE_NONFREE=ON
```
With CUDA:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..\
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=ON \
    -D OPENCV_DNN_CUDA=ON
    -D WITH_CUDNN=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=8.6 \
    -D WITH_CUBLAS=1 
```
CUDA_ARCH_BIN can be found here: https://developer.nvidia.com/cuda-gpus

5. Compilation process:
```
make -j $(expr $(nproc) / 2)
```

6. Installation process:
```
sudo make install
```

7. Import OpenCV package:
```
pkg-config opencv4 --cflags --libs
```

8. Add in *~/.bashrc*:
```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/' >> ~/.bashrc
```

9. Load the libraries into the current shell:
```
source ~/.bashrc
```

## PCL installation from source (do not install without supervision)

The best way to be updated is to install PCL compiling from the source: https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html

It fixes some problems when you try to use Ubuntu 22.04, VTK 9.1, and PCL 1.12.1

1. Create a folder:
```
mkdir ~/pcl_build && cd ~/pcl_build
```
2. Download a [Stable version] (latest recommended) and uncompress:
```
tar xvf pcl-pcl-<version>.tar.gz
```
or clone PCL repository (experimental):
```
git clone --recursive https://github.com/PointCloudLibrary/pcl.git
```

3. Create a temporary build directory:
```
cd ~/pcl_build/<pcl-folder>
mkdir build && cd build
```

4. Run the CMake build system using the default options:
```
cmake ..
```
Please note that cmake might default to a debug build. If you want to compile a release build of PCL with enhanced compiler optimizations, you can change the build target to “Release” with “-DCMAKE_BUILD_TYPE=Release”:
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```

5. Compilation process (if it fails, rerun the command):
```
make -j2
```

6. Installation process:
```
sudo make -j2 install
```

## About

This project was made by [José Miguel Guerrero], Associate Professor at [Universidad Rey Juan Carlos].

Copyright &copy; 2024.

[![Twitter](https://img.shields.io/badge/follow-@jm__guerrero-green.svg)](https://twitter.com/jm__guerrero)

## License

This work is licensed under the terms of the [MIT license](https://opensource.org/license/mit).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Universidad Rey Juan Carlos]: https://www.urjc.es/
[José Miguel Guerrero]: https://sites.google.com/view/jmguerrero
[Stable version]: https://github.com/PointCloudLibrary/pcl/releases

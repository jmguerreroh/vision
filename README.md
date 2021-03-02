[![License badge](https://img.shields.io/badge/license-Apache2-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Twitter](https://img.shields.io/badge/follow-@jm__guerrero-green.svg)](https://twitter.com/jm__guerrero)

# Computer Vision Examples

This project contains code examples created un Visual Studio Code for Computer Vision using C++ & OpenCV. These examples are created for the Computer Vision Subject of Robotics Software Engineering Degree at URJC.

## Installation

Open Ubuntu terminal and follow the next steps:

1. Install dependencies:
```
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev
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
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
```

5. Compilation process:
```
make -j8
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
export LD_LIBRARY_PATH=$LS_LIBRARY_PATH:/usr/local/lib/
```

9. Load the libraries into the current shell:
```
source ~/.bashrc
```


## About

This is a project made by [José Miguel Guerrero], Assistant Professor at [Universidad Rey Juan Carlos].
Copyright &copy; 2021.

[Universidad Rey Juan Carlos]: https://www.urjc.es/
[José Miguel Guerrero]: https://sites.google.com/view/jmguerrero
# Makefile for Discrete Fourier Transform (DFT) example
# Author: José Miguel Guerrero Hernández

# Compiler settings
CXX       = g++
CXXFLAGS  = -Wall -Wextra -std=c++17 -O2

# OpenCV configuration (using pkg-config)
OPENCV    = $(shell pkg-config opencv4 --cflags --libs)

# Source files and targets
SRC1     = main.cpp
# Both executables take the name of the folder, matching what the top-level
# CMakeLists.txt generates: <folder> for main.cpp, <folder>_<stem> otherwise
TARGET1  = $(notdir $(CURDIR))

SRC2     = frequencies.cpp
TARGET2  = $(notdir $(CURDIR))_frequencies

# Default target: build both executables
all: $(TARGET1) $(TARGET2)

# Build main.cpp
$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

# Build frequencies.cpp
$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

# Clean build artifacts
clean:
	rm -f $(TARGET1) $(TARGET2)

# Declare phony targets (not actual files)
.PHONY: all clean
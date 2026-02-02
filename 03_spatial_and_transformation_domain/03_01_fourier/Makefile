# Makefile for Discrete Fourier Transform (DFT) example
# Author: José Miguel Guerrero Hernández

# Compiler settings
CXX       = g++
CXXFLAGS  = -Wall -Wextra -std=c++17 -O2

# OpenCV configuration (using pkg-config)
OPENCV    = $(shell pkg-config opencv4 --cflags --libs)

# Source files and targets
SRC1     = main.cpp
TARGET1  = fourier

SRC2     = frequencies.cpp
TARGET2  = frequencies

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
# Makefile for YOLO11 (Ultralytics) example
# Author: José Miguel Guerrero Hernández

# Compiler settings
CXX       = g++
CXXFLAGS  = -Wall -Wextra -std=c++17 -O2

# OpenCV configuration (using pkg-config)
OPENCV    = $(shell pkg-config opencv4 --cflags --libs)

# Source file and target
SRC    = main.cpp
TARGET = yolo11

# Default target: build executable
all: $(TARGET)

# Build main.cpp
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread $(OPENCV)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Declare phony targets (not actual files)
.PHONY: all clean

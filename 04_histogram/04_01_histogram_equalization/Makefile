# Makefile for Histogram Equalization example
# Author: José Miguel Guerrero Hernández

# Compiler settings
CXX       = g++
CXXFLAGS  = -Wall -Wextra -std=c++17 -O2

# OpenCV configuration (using pkg-config)
OPENCV    = $(shell pkg-config opencv4 --cflags --libs)

# Source file and target
SRC    = main.cpp
# The executable takes the name of the folder, the same one the top-level
# CMakeLists.txt gives it, so both ways of building produce the same binary
TARGET = $(notdir $(CURDIR))

# Default target: build executable
all: $(TARGET)

# Build main.cpp
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Declare phony targets (not actual files)
.PHONY: all clean
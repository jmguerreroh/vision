#!/usr/bin/env python3
"""Generate test image with simple shapes for chain code demonstration"""

import cv2
import numpy as np

# Create white canvas (2x2 grid)
grid_size = 2
cell_size = 300
margin = 50
img_size = grid_size * cell_size + (grid_size + 1) * margin

img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

# Calculate cell centers
def get_cell_center(row, col):
    cx = margin + col * (cell_size + margin) + cell_size // 2
    cy = margin + row * (cell_size + margin) + cell_size // 2
    return cx, cy

# Shape size (same for all)
shape_size = 100

# Cell 0,0: Rectangle (top-left)
cx, cy = get_cell_center(0, 0)
cv2.rectangle(img, 
              (cx - shape_size//2, cy - shape_size//2), 
              (cx + shape_size//2, cy + shape_size//2), 
              (0, 0, 0), -1)

# Cell 0,1: Circle (top-right)
cx, cy = get_cell_center(0, 1)
cv2.circle(img, (cx, cy), shape_size//2, (0, 0, 0), -1)

# Cell 1,0: Triangle (bottom-left)
cx, cy = get_cell_center(1, 0)
h = int(shape_size * 0.866)  # height of equilateral triangle
triangle = np.array([
    [cx, cy - h//2],              # top
    [cx - shape_size//2, cy + h//2],  # bottom-left
    [cx + shape_size//2, cy + h//2]   # bottom-right
], np.int32)
cv2.fillPoly(img, [triangle], (0, 0, 0))

# Cell 1,1: Square rotated 45 degrees (diamond) (bottom-right)
cx, cy = get_cell_center(1, 1)
d = int(shape_size / 1.414)  # diagonal/sqrt(2) for 45° rotation
square = np.array([
    [cx, cy - d],      # top
    [cx - d, cy],      # left
    [cx, cy + d],      # bottom
    [cx + d, cy]       # right
], np.int32)
cv2.fillPoly(img, [square], (0, 0, 0))

# Save image
cv2.imwrite('shapes.png', img)
print(f"Created shapes.png ({img_size}x{img_size})")
print(f"Layout: 2x2 grid, cell size: {cell_size}x{cell_size}, margin: {margin}px")
print("Shapes: [Rectangle, Circle] / [Triangle, Diamond]")

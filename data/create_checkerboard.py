#!/usr/bin/env python3
"""Generate a checkerboard pattern for corner detection testing"""

import cv2
import numpy as np

# Create checkerboard pattern
rows, cols = 8, 8
square_size = 60
img_height = rows * square_size
img_width = cols * square_size

# Create white image
checkerboard = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

# Draw black squares
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            y1 = i * square_size
            y2 = (i + 1) * square_size
            x1 = j * square_size
            x2 = (j + 1) * square_size
            checkerboard[y1:y2, x1:x2] = 0

# Save
cv2.imwrite('checkerboard.png', checkerboard)
print(f"Created checkerboard.png ({img_width}x{img_height})")
print(f"Pattern: {rows}x{cols} squares, {square_size}x{square_size} pixels each")

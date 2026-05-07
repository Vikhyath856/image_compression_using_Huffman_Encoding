"""
Generate sample test images for quick demos.
Run: python generate_samples.py
"""

import numpy as np
from PIL import Image
from pathlib import Path

out = Path(__file__).parent
out.mkdir(exist_ok=True)

# Grayscale gradient
arr = np.tile(np.arange(256, dtype=np.uint8), (128, 2))
Image.fromarray(arr, "L").save(out / "gradient_gray.png")
print("Created: gradient_gray.png")

# Random grayscale noise
arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
Image.fromarray(arr, "L").save(out / "noise_gray.png")
print("Created: noise_gray.png")

# Random RGB
arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
Image.fromarray(arr, "RGB").save(out / "noise_rgb.png")
print("Created: noise_rgb.png")

# Uniform (best case for Huffman)
arr = np.full((64, 64), 200, dtype=np.uint8)
Image.fromarray(arr, "L").save(out / "uniform.png")
print("Created: uniform.png")

print("\nAll sample images saved to samples/")

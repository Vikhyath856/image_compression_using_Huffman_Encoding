# Image Compression using Huffman Coding

**Author:** Vikhyath B M  
**Tech Stack:** Python · NumPy · Pillow · Matplotlib

---

## Overview

This project implements **lossless image compression** using **Huffman Encoding** — a classic greedy algorithm from information theory. Pixel values are assigned variable-length binary codes based on their frequency: frequent values get shorter codes, rare values get longer ones.

The result is a compressed `.huff` binary file that can be **perfectly decoded** back to the original image with zero data loss.

---

## Project Structure

```
image_compression_huffman/
├── src/
│   ├── huffman.py       # Core: tree building, code generation, encode/decode
│   ├── compress.py      # Image I/O, compression pipeline, file format
│   ├── visualize.py     # Plots: frequency, code lengths, tree, before/after
│   └── main.py          # CLI entry point
├── tests/
│   └── test_huffman.py  # Unit + integration tests (pytest)
├── demo_output/         # Auto-created by demo command
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

All commands are run from the `src/` directory:

```bash
cd src
```

### Compress an image
```bash
python main.py compress path/to/image.png output.huff
```

### Decompress back to image
```bash
python main.py decompress output.huff restored.png
```

### Full Demo (compress + decompress + visualize)
```bash
python main.py demo path/to/image.png
```

This generates in `demo_output/`:
| File | Description |
|------|-------------|
| `*.huff` | Compressed binary file |
| `*_decompressed.png` | Restored image |
| `freq_distribution.png` | Pixel value frequency bar chart |
| `code_lengths.png` | Huffman code length histogram |
| `before_after.png` | Side-by-side lossless verification |
| `huffman_tree.png` | Tree structure (top-16 symbols) |

---

## Run Tests

```bash
# From project root
pytest tests/ -v
```

Tests cover:
- Frequency table construction
- Huffman tree properties (prefix-free codes, frequency ordering)
- Encode/decode round-trip correctness
- Full compress → decompress → pixel-equality pipeline
- Edge cases: uniform images, single-symbol, small images

---

## How It Works

1. **Frequency Analysis** — Count occurrences of each pixel value (0–255).
2. **Build Min-Heap** — Each pixel value becomes a leaf node with its frequency.
3. **Construct Tree** — Repeatedly merge the two lowest-frequency nodes until one root remains.
4. **Assign Codes** — Traverse the tree: left edge = `0`, right edge = `1`. Each leaf gets a unique binary code.
5. **Encode** — Replace every pixel byte with its Huffman code → get a long bit string.
6. **Pack & Save** — Convert bit string to bytes and write with metadata (tree frequencies, image shape, padding) to a `.huff` file.
7. **Decode** — Read metadata, rebuild the tree, traverse bit-by-bit to recover original pixel values.

---

## Compression Metrics

| Metric | Description |
|--------|-------------|
| **Compression Ratio** | `original_size / compressed_size` |
| **Space Saved %** | `(1 - compressed/original) × 100` |
| **Avg Code Length** | Weighted average bits per symbol |
| **Lossless** | Pixel-exact reconstruction verified by `np.array_equal` |

---

## Sample Results

Typical results on natural images:

| Image Type | Original | Compressed | Ratio |
|------------|----------|------------|-------|
| Grayscale photo | 65 KB | ~52 KB | ~1.25× |
| RGB photo | 192 KB | ~155 KB | ~1.24× |
| Uniform color image | 4 KB | ~0.1 KB | ~40× |

> Note: Huffman coding alone is less aggressive than formats like PNG (which also uses LZ77). It is most effective on images with highly skewed pixel distributions.

---

## References

- Huffman, D. A. (1952). *A Method for the Construction of Minimum-Redundancy Codes.* Proceedings of the IRE.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory.* Wiley.

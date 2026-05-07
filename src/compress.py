"""
Image Compression Pipeline using Huffman Coding
Author: Vikhyath B M
Description: Handles image I/O, compression, and decompression using Huffman coding.
"""

import pickle
import struct
import numpy as np
from pathlib import Path
from PIL import Image

from huffman import (
    build_frequency_table,
    build_huffman_tree,
    generate_codes,
    encode,
    decode,
    bit_string_to_bytes,
    bytes_to_bit_string,
    calculate_compression_ratio,
    calculate_average_code_length,
)


def load_image(path: str) -> tuple[np.ndarray, str]:
    """
    Load an image and return its pixel array and mode.

    Args:
        path: Path to image file

    Returns:
        (pixel_array, mode) — mode is 'L' (grayscale) or 'RGB'
    """
    img = Image.open(path)
    mode = img.mode
    if mode not in ("L", "RGB"):
        img = img.convert("RGB")
        mode = "RGB"
    return np.array(img), mode


def save_image(pixel_array: np.ndarray, mode: str, path: str):
    """Save a pixel array as an image file."""
    img = Image.fromarray(pixel_array.astype(np.uint8), mode)
    img.save(path)


def compress_image(input_path: str, output_path: str) -> dict:
    """
    Compress an image using Huffman coding and save to a .huff file.

    Args:
        input_path: Path to original image
        output_path: Path to save compressed .huff file

    Returns:
        Dictionary with compression statistics
    """
    pixels, mode = load_image(input_path)
    shape = pixels.shape
    flat_data = pixels.flatten().tobytes()
    original_size = len(flat_data)

    # Build Huffman tree
    freq_table = build_frequency_table(flat_data)
    tree = build_huffman_tree(freq_table)
    codes = generate_codes(tree)

    # Encode data
    bit_string, padding = encode(flat_data, codes)
    compressed_bytes = bit_string_to_bytes(bit_string)

    # Package metadata + compressed data
    metadata = {
        "freq_table": freq_table,
        "shape": shape,
        "mode": mode,
        "padding": padding,
        "original_length": len(flat_data),
    }

    with open(output_path, "wb") as f:
        meta_bytes = pickle.dumps(metadata)
        # Write metadata length (4 bytes) then metadata, then compressed data
        f.write(struct.pack(">I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(compressed_bytes)

    compressed_size = Path(output_path).stat().st_size

    stats = {
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": calculate_compression_ratio(original_size, compressed_size),
        "space_saved_percent": round((1 - compressed_size / original_size) * 100, 2),
        "average_code_length": round(calculate_average_code_length(codes, freq_table), 4),
        "unique_symbols": len(freq_table),
        "image_shape": shape,
        "mode": mode,
    }

    return stats


def decompress_image(input_path: str, output_path: str) -> np.ndarray:
    """
    Decompress a .huff file back to an image.

    Args:
        input_path: Path to .huff compressed file
        output_path: Path to save decompressed image

    Returns:
        Reconstructed pixel array
    """
    with open(input_path, "rb") as f:
        meta_len = struct.unpack(">I", f.read(4))[0]
        metadata = pickle.loads(f.read(meta_len))
        compressed_bytes = f.read()

    freq_table = metadata["freq_table"]
    shape = metadata["shape"]
    mode = metadata["mode"]
    padding = metadata["padding"]
    original_length = metadata["original_length"]

    # Rebuild tree and decode
    tree = build_huffman_tree(freq_table)
    bit_string = bytes_to_bit_string(compressed_bytes)

    # Remove padding bits
    if padding > 0:
        bit_string = bit_string[:-padding]

    decoded_bytes = decode(bit_string, tree, original_length)
    pixels = np.frombuffer(decoded_bytes, dtype=np.uint8).reshape(shape)

    save_image(pixels, mode, output_path)
    return pixels


def verify_lossless(original_path: str, decompressed_path: str) -> bool:
    """
    Verify the decompressed image is pixel-perfect (lossless).

    Returns:
        True if identical, False otherwise
    """
    orig = np.array(Image.open(original_path))
    decomp = np.array(Image.open(decompressed_path))
    return np.array_equal(orig, decomp)

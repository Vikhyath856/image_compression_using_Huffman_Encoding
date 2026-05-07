"""
Unit Tests — Huffman Image Compression
Author: Vikhyath B M
"""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
from compress import compress_image, decompress_image, verify_lossless


# ─── Helpers ────────────────────────────────────────────────────────────────────

def make_test_image(size=(64, 64), mode="L") -> str:
    """Create a temporary test image and return its path."""
    arr = np.random.randint(0, 256, (*size, 3) if mode == "RGB" else size, dtype=np.uint8)
    img = Image.fromarray(arr, mode)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


# ─── Huffman Core Tests ──────────────────────────────────────────────────────────

class TestFrequencyTable:
    def test_basic_count(self):
        data = bytes([0, 1, 1, 2, 2, 2])
        freq = build_frequency_table(data)
        assert freq[0] == 1
        assert freq[1] == 2
        assert freq[2] == 3

    def test_single_symbol(self):
        data = bytes([5] * 10)
        freq = build_frequency_table(data)
        assert freq == {5: 10}

    def test_all_unique(self):
        data = bytes(range(256))
        freq = build_frequency_table(data)
        assert len(freq) == 256
        assert all(v == 1 for v in freq.values())


class TestHuffmanTree:
    def test_tree_builds(self):
        freq = {0: 5, 1: 3, 2: 1}
        root = build_huffman_tree(freq)
        assert root is not None
        assert root.freq == 9

    def test_single_symbol_tree(self):
        freq = {42: 100}
        root = build_huffman_tree(freq)
        assert root.is_leaf()
        assert root.symbol == 42

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_huffman_tree({})


class TestCodeGeneration:
    def test_codes_cover_all_symbols(self):
        freq = {0: 10, 1: 5, 2: 3, 3: 1}
        tree = build_huffman_tree(freq)
        codes = generate_codes(tree)
        assert set(codes.keys()) == {0, 1, 2, 3}

    def test_codes_are_prefix_free(self):
        freq = build_frequency_table(bytes(range(50)) * 10)
        tree = build_huffman_tree(freq)
        codes = generate_codes(tree)
        code_list = list(codes.values())
        for i, c1 in enumerate(code_list):
            for j, c2 in enumerate(code_list):
                if i != j:
                    assert not c1.startswith(c2) or c1 == c2, f"{c1} is prefix of {c2}"

    def test_higher_freq_shorter_code(self):
        freq = {0: 100, 1: 1}
        tree = build_huffman_tree(freq)
        codes = generate_codes(tree)
        assert len(codes[0]) <= len(codes[1])


class TestEncodeDecodeRoundtrip:
    def test_roundtrip_simple(self):
        data = bytes([0, 1, 2, 0, 1, 0, 0])
        freq = build_frequency_table(data)
        tree = build_huffman_tree(freq)
        codes = generate_codes(tree)
        bit_string, padding = encode(data, codes)
        if padding > 0:
            bit_string_trimmed = bit_string[:-padding]
        else:
            bit_string_trimmed = bit_string
        decoded = decode(bit_string_trimmed, tree, len(data))
        assert decoded == data

    def test_roundtrip_random(self):
        data = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        freq = build_frequency_table(data)
        tree = build_huffman_tree(freq)
        codes = generate_codes(tree)
        bit_string, padding = encode(data, codes)
        compressed = bit_string_to_bytes(bit_string)
        recovered = bytes_to_bit_string(compressed)
        if padding > 0:
            recovered = recovered[:-padding]
        decoded = decode(recovered, tree, len(data))
        assert decoded == data


class TestMetrics:
    def test_compression_ratio(self):
        assert calculate_compression_ratio(100, 50) == pytest.approx(2.0)
        assert calculate_compression_ratio(100, 100) == pytest.approx(1.0)

    def test_average_code_length(self):
        freq = {0: 1, 1: 1}
        tree = build_huffman_tree(freq)
        codes = generate_codes(tree)
        avg = calculate_average_code_length(codes, freq)
        assert avg == pytest.approx(1.0)


# ─── Integration Tests ───────────────────────────────────────────────────────────

class TestCompressionPipeline:
    def test_compress_decompress_grayscale(self, tmp_path):
        src = make_test_image(size=(64, 64), mode="L")
        compressed = str(tmp_path / "test.huff")
        decompressed = str(tmp_path / "test_out.png")

        stats = compress_image(src, compressed)
        assert stats["original_size_bytes"] > 0
        assert stats["compressed_size_bytes"] > 0

        decompress_image(compressed, decompressed)
        assert verify_lossless(src, decompressed)
        os.unlink(src)

    def test_compress_decompress_rgb(self, tmp_path):
        src = make_test_image(size=(32, 32), mode="RGB")
        compressed = str(tmp_path / "test_rgb.huff")
        decompressed = str(tmp_path / "test_rgb_out.png")

        compress_image(src, compressed)
        decompress_image(compressed, decompressed)
        assert verify_lossless(src, decompressed)
        os.unlink(src)

    def test_small_image(self, tmp_path):
        src = make_test_image(size=(4, 4), mode="L")
        compressed = str(tmp_path / "small.huff")
        decompressed = str(tmp_path / "small_out.png")

        compress_image(src, compressed)
        decompress_image(compressed, decompressed)
        assert verify_lossless(src, decompressed)
        os.unlink(src)

    def test_uniform_image(self, tmp_path):
        """Image with single pixel value — edge case."""
        arr = np.full((32, 32), 128, dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        src = str(tmp_path / "uniform.png")
        img.save(src)

        compressed = str(tmp_path / "uniform.huff")
        decompressed = str(tmp_path / "uniform_out.png")

        compress_image(src, compressed)
        decompress_image(compressed, decompressed)
        assert verify_lossless(src, decompressed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

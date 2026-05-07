"""
Huffman Coding Implementation for Image Compression
Author: Vikhyath B M
Description: Core Huffman tree construction, encoding, and decoding logic.
"""

import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class HuffmanNode:
    """A node in the Huffman tree."""
    freq: int
    symbol: Optional[int] = field(default=None, compare=False)
    left: Optional["HuffmanNode"] = field(default=None, compare=False)
    right: Optional["HuffmanNode"] = field(default=None, compare=False)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_frequency_table(data: bytes) -> dict:
    """
    Build a frequency table from raw byte data.

    Args:
        data: Raw bytes (e.g., flattened pixel array)

    Returns:
        Dictionary mapping byte value -> frequency
    """
    return dict(Counter(data))


def build_huffman_tree(freq_table: dict) -> HuffmanNode:
    """
    Build Huffman tree from frequency table using a min-heap.

    Args:
        freq_table: {symbol: frequency} mapping

    Returns:
        Root HuffmanNode of the tree
    """
    if not freq_table:
        raise ValueError("Frequency table is empty.")

    heap = [HuffmanNode(freq=freq, symbol=sym) for sym, freq in freq_table.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(root: HuffmanNode) -> dict:
    """
    Traverse the Huffman tree and generate binary codes for each symbol.

    Args:
        root: Root of the Huffman tree

    Returns:
        Dictionary mapping symbol -> binary code string (e.g., '010')
    """
    codes = {}

    def _traverse(node: HuffmanNode, current_code: str):
        if node is None:
            return
        if node.is_leaf():
            codes[node.symbol] = current_code if current_code else "0"
            return
        _traverse(node.left, current_code + "0")
        _traverse(node.right, current_code + "1")

    _traverse(root, "")
    return codes


def encode(data: bytes, codes: dict) -> tuple[str, int]:
    """
    Encode byte data using Huffman codes.

    Args:
        data: Original bytes
        codes: Symbol -> code string mapping

    Returns:
        (bit_string, padding) tuple
    """
    bit_string = "".join(codes[byte] for byte in data)
    # Pad to make it a multiple of 8
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    return bit_string, padding


def bit_string_to_bytes(bit_string: str) -> bytes:
    """Convert a binary string to actual bytes."""
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_array.append(int(bit_string[i:i + 8], 2))
    return bytes(byte_array)


def bytes_to_bit_string(data: bytes) -> str:
    """Convert bytes back to a binary string."""
    return "".join(f"{byte:08b}" for byte in data)


def decode(bit_string: str, root: HuffmanNode, original_length: int) -> bytes:
    """
    Decode a Huffman-encoded bit string back to bytes.

    Args:
        bit_string: Encoded binary string (may include padding)
        root: Root of the Huffman tree used for encoding
        original_length: Number of original symbols to decode

    Returns:
        Decoded bytes
    """
    decoded = bytearray()
    node = root
    count = 0

    for bit in bit_string:
        node = node.left if bit == "0" else node.right
        if node.is_leaf():
            decoded.append(node.symbol)
            node = root
            count += 1
            if count == original_length:
                break

    return bytes(decoded)


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Return compression ratio (original / compressed)."""
    return original_size / compressed_size if compressed_size else float("inf")


def calculate_average_code_length(codes: dict, freq_table: dict) -> float:
    """Calculate the weighted average code length."""
    total_symbols = sum(freq_table.values())
    return sum(len(codes[sym]) * freq for sym, freq in freq_table.items()) / total_symbols

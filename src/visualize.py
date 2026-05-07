"""
Visualization Utilities
Author: Vikhyath B M
Description: Plot Huffman tree, frequency distribution, and compression statistics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from huffman import HuffmanNode, build_huffman_tree, generate_codes, build_frequency_table


# ─── Frequency Distribution ────────────────────────────────────────────────────

def plot_frequency_distribution(freq_table: dict, title: str = "Pixel Frequency Distribution", save_path: str = None):
    """Bar chart of pixel value frequencies."""
    symbols = sorted(freq_table.keys())
    freqs = [freq_table[s] for s in symbols]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(symbols, freqs, width=1, color="steelblue", edgecolor="none")
    ax.set_xlabel("Pixel Value (0–255)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xlim(-1, 256)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved frequency plot → {save_path}")
    else:
        plt.show()
    plt.close()


# ─── Code Length Distribution ───────────────────────────────────────────────────

def plot_code_lengths(codes: dict, save_path: str = None):
    """Show distribution of Huffman code lengths."""
    lengths = [len(code) for code in codes.values()]
    max_len = max(lengths)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=range(1, max_len + 2), color="coral", edgecolor="white", align="left")
    ax.set_xlabel("Code Length (bits)")
    ax.set_ylabel("Number of Symbols")
    ax.set_title("Huffman Code Length Distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved code length plot → {save_path}")
    else:
        plt.show()
    plt.close()


# ─── Before / After Comparison ─────────────────────────────────────────────────

def plot_before_after(original_path: str, decompressed_path: str, save_path: str = None):
    """Side-by-side comparison of original and decompressed image."""
    orig = np.array(Image.open(original_path))
    decomp = np.array(Image.open(decompressed_path))
    identical = np.array_equal(orig, decomp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig, cmap="gray" if orig.ndim == 2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(decomp, cmap="gray" if decomp.ndim == 2 else None)
    axes[1].set_title(f"Decompressed Image\n({'✅ Lossless' if identical else '❌ Mismatch'})")
    axes[1].axis("off")

    plt.suptitle("Huffman Compression — Lossless Verification", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison plot → {save_path}")
    else:
        plt.show()
    plt.close()


# ─── Compression Stats Summary ─────────────────────────────────────────────────

def print_stats(stats: dict):
    """Pretty-print compression statistics."""
    print("\n" + "=" * 50)
    print("  HUFFMAN COMPRESSION STATISTICS")
    print("=" * 50)
    print(f"  Image Shape          : {stats['image_shape']}")
    print(f"  Mode                 : {stats['mode']}")
    print(f"  Original Size        : {stats['original_size_bytes']:,} bytes")
    print(f"  Compressed Size      : {stats['compressed_size_bytes']:,} bytes")
    print(f"  Compression Ratio    : {stats['compression_ratio']:.3f}x")
    print(f"  Space Saved          : {stats['space_saved_percent']}%")
    print(f"  Avg Code Length      : {stats['average_code_length']} bits/symbol")
    print(f"  Unique Symbols       : {stats['unique_symbols']}")
    print("=" * 50 + "\n")


# ─── Huffman Tree Visualizer ────────────────────────────────────────────────────

def _get_tree_positions(node: HuffmanNode, x=0, y=0, dx=1.0, positions=None, edges=None):
    if positions is None:
        positions = {}
        edges = []

    node_id = id(node)
    positions[node_id] = (x, y, node)

    if node.left:
        edges.append((node_id, id(node.left), "0"))
        _get_tree_positions(node.left, x - dx, y - 1, dx / 2, positions, edges)
    if node.right:
        edges.append((node_id, id(node.right), "1"))
        _get_tree_positions(node.right, x + dx, y - 1, dx / 2, positions, edges)

    return positions, edges


def visualize_huffman_tree(freq_table: dict, max_symbols: int = 16, save_path: str = None):
    """
    Visualize the Huffman tree (limited to top-N most frequent symbols for clarity).
    """
    # Limit to top N symbols
    top = dict(sorted(freq_table.items(), key=lambda x: -x[1])[:max_symbols])
    tree = build_huffman_tree(top)
    positions, edges = _get_tree_positions(tree, dx=max_symbols / 2)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title(f"Huffman Tree (Top {max_symbols} symbols by frequency)", fontsize=13)

    # Draw edges
    for parent_id, child_id, label in edges:
        px, py, _ = positions[parent_id]
        cx, cy, _ = positions[child_id]
        ax.plot([px, cx], [py, cy], "k-", lw=0.8, zorder=1)
        mx, my = (px + cx) / 2, (py + cy) / 2
        ax.text(mx, my, label, fontsize=7, ha="center", color="crimson", zorder=3)

    # Draw nodes
    for node_id, (x, y, node) in positions.items():
        color = "#4CAF50" if node.is_leaf() else "#2196F3"
        circle = plt.Circle((x, y), 0.25, color=color, zorder=2)
        ax.add_patch(circle)
        label = str(node.symbol) if node.is_leaf() else str(node.freq)
        ax.text(x, y, label, ha="center", va="center", fontsize=6, color="white", zorder=4, fontweight="bold")

    leaf_patch = mpatches.Patch(color="#4CAF50", label="Leaf (pixel value)")
    int_patch = mpatches.Patch(color="#2196F3", label="Internal node (freq)")
    ax.legend(handles=[leaf_patch, int_patch], loc="upper right")
    ax.axis("off")
    ax.autoscale()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved Huffman tree → {save_path}")
    else:
        plt.show()
    plt.close()

"""
Image Compression using Huffman Coding — CLI Entry Point
Author: Vikhyath B M

Usage:
    python main.py compress   <input_image> <output.huff>
    python main.py decompress <input.huff>  <output_image>
    python main.py demo       <input_image>
"""

import sys
import argparse
from pathlib import Path

from compress import compress_image, decompress_image, verify_lossless, load_image
from visualize import (
    plot_frequency_distribution,
    plot_code_lengths,
    plot_before_after,
    print_stats,
    visualize_huffman_tree,
)
from huffman import (
    build_frequency_table,
    build_huffman_tree,
    generate_codes,
)


def cmd_compress(args):
    print(f"\n🔵 Compressing: {args.input} → {args.output}")
    stats = compress_image(args.input, args.output)
    print_stats(stats)


def cmd_decompress(args):
    print(f"\n🟢 Decompressing: {args.input} → {args.output}")
    decompress_image(args.input, args.output)
    print(f"✅ Decompressed image saved to: {args.output}")


def cmd_demo(args):
    """
    Full end-to-end demo:
      1. Compress → Decompress → Verify
      2. Generate all visualizations
    """
    input_path = args.input
    stem = Path(input_path).stem
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    compressed_path = str(output_dir / f"{stem}.huff")
    decompressed_path = str(output_dir / f"{stem}_decompressed.png")

    print(f"\n{'='*55}")
    print(f"  DEMO: Huffman Image Compression")
    print(f"  Input : {input_path}")
    print(f"{'='*55}\n")

    # Step 1: Compress
    print("📦 Step 1: Compressing...")
    stats = compress_image(input_path, compressed_path)
    print_stats(stats)

    # Step 2: Decompress
    print("📂 Step 2: Decompressing...")
    decompress_image(compressed_path, decompressed_path)

    # Step 3: Verify
    print("🔍 Step 3: Verifying lossless integrity...")
    identical = verify_lossless(input_path, decompressed_path)
    print(f"   Result: {'✅ LOSSLESS — Pixel-perfect match!' if identical else '❌ Mismatch detected!'}\n")

    # Step 4: Visualizations
    print("📊 Step 4: Generating visualizations...")
    pixels, mode = load_image(input_path)
    flat = pixels.flatten().tobytes()
    freq_table = build_frequency_table(flat)
    tree = build_huffman_tree(freq_table)
    codes = generate_codes(tree)

    plot_frequency_distribution(freq_table, save_path=str(output_dir / "freq_distribution.png"))
    plot_code_lengths(codes, save_path=str(output_dir / "code_lengths.png"))
    plot_before_after(input_path, decompressed_path, save_path=str(output_dir / "before_after.png"))
    visualize_huffman_tree(freq_table, save_path=str(output_dir / "huffman_tree.png"))

    print(f"\n✅ Demo complete! All outputs saved to: {output_dir}/")
    print("   Files generated:")
    for f in sorted(output_dir.iterdir()):
        print(f"   - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Image Compression using Huffman Coding | Vikhyath B M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # compress
    p_compress = subparsers.add_parser("compress", help="Compress an image to .huff")
    p_compress.add_argument("input", help="Input image path (PNG, JPG, BMP, etc.)")
    p_compress.add_argument("output", help="Output .huff file path")
    p_compress.set_defaults(func=cmd_compress)

    # decompress
    p_decompress = subparsers.add_parser("decompress", help="Decompress a .huff file to an image")
    p_decompress.add_argument("input", help="Input .huff file path")
    p_decompress.add_argument("output", help="Output image path (PNG recommended)")
    p_decompress.set_defaults(func=cmd_decompress)

    # demo
    p_demo = subparsers.add_parser("demo", help="Full compress+decompress+visualize demo")
    p_demo.add_argument("input", help="Input image path")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

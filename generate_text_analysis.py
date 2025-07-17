#!/usr/bin/env python3
"""
Generate text-based analysis from saved plot data.

This script loads a plot_data.pkl file and generates both human-readable text
summaries and structured JSON analysis optimized for language model consumption.

Usage:
    python generate_text_analysis.py <plot_data.pkl> [--output-dir <dir>]

Examples:
    # Generate analysis in same directory as pkl file
    python generate_text_analysis.py logs/20250717-155819/plots/plot_data.pkl
    
    # Generate analysis in custom directory
    python generate_text_analysis.py data.pkl --output-dir analysis_results/
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_analysis import save_text_analysis


def main():
    parser = argparse.ArgumentParser(description="Generate text analysis from plot data")
    parser.add_argument("plot_data_file", help="Path to plot_data.pkl file")
    parser.add_argument("--output-dir", help="Output directory (default: same as pkl file)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.plot_data_file):
        print(f"Error: File not found: {args.plot_data_file}")
        sys.exit(1)
    
    if not args.plot_data_file.endswith('.pkl'):
        print(f"Warning: Expected .pkl file, got: {args.plot_data_file}")
    
    try:
        # Generate text analysis
        text_path, json_path = save_text_analysis(args.plot_data_file, args.output_dir)
        
        print(f"✓ Generated text analysis: {text_path}")
        print(f"✓ Generated JSON analysis: {json_path}")
        
        # Show quick summary
        print(f"\nQuick Summary:")
        print(f"Text analysis: {os.path.getsize(text_path)} bytes")
        print(f"JSON analysis: {os.path.getsize(json_path)} bytes")
        
        # Show first few lines of text analysis
        print(f"\nFirst few lines of text analysis:")
        print("-" * 50)
        with open(text_path, 'r') as f:
            lines = f.readlines()[:10]
            for line in lines:
                print(line.rstrip())
        if len(lines) >= 10:
            print("...")
        
    except Exception as e:
        print(f"Error generating text analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
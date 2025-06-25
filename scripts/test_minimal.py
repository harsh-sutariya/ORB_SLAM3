#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', required=True)
    parser.add_argument('--right_file', required=True)
    parser.add_argument('--out_dir', default='./test_outputs/')
    args = parser.parse_args()
    
    print(f"Python script called successfully!")
    print(f"Left image: {args.left_file}")
    print(f"Right image: {args.right_file}")
    print(f"Output dir: {args.out_dir}")
    
    # Check if files exist
    if not os.path.exists(args.left_file):
        print(f"Error: Left image not found: {args.left_file}")
        return 1
        
    if not os.path.exists(args.right_file):
        print(f"Error: Right image not found: {args.right_file}")
        return 1
    
    # Try to load images
    try:
        left_img = Image.open(args.left_file)
        right_img = Image.open(args.right_file)
        print(f"Left image size: {left_img.size}")
        print(f"Right image size: {right_img.size}")
        
        # Create dummy disparity (for testing integration)
        width, height = left_img.size
        dummy_disparity = np.random.rand(height, width) * 50  # Random disparity values
        
        # Save dummy disparity
        os.makedirs(args.out_dir, exist_ok=True)
        disparity_path = os.path.join(args.out_dir, 'disparity.tiff')
        
        # Save as 16-bit TIFF
        disparity_16bit = (dummy_disparity * 256).astype(np.uint16)
        Image.fromarray(disparity_16bit).save(disparity_path)
        
        print(f"Dummy disparity saved to: {disparity_path}")
        print("Integration test successful!")
        
    except Exception as e:
        print(f"Error processing images: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
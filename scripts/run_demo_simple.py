#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from omegaconf import OmegaConf
    from core.utils.utils import InputPadder
    from core.foundation_stereo import FoundationStereo
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the required packages installed and paths are correct")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='FoundationStereo Demo (Simplified)')
    parser.add_argument('--left_file', type=str, required=True, help='Left image path')
    parser.add_argument('--right_file', type=str, required=True, help='Right image path')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--out_dir', type=str, default='./outputs/', help='Output directory')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--valid_iters', type=int, default=32, help='Number of validation iterations')
    return parser.parse_args()

def load_image(image_file):
    """Load and prepare image for processing"""
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()  # Add batch dimension and move to GPU

def save_disparity(disparity, output_path):
    """Save disparity map in multiple formats"""
    # Convert to numpy
    disp_np = disparity.squeeze().cpu().numpy()
    
    # Save as TIFF (16-bit for precision)
    disp_16bit = (disp_np * 256).astype(np.uint16)
    cv2.imwrite(output_path.replace('.png', '.tiff'), disp_16bit)
    
    # Save as PNG (8-bit for visualization)
    disp_norm = (disp_np / disp_np.max() * 255).astype(np.uint8)
    cv2.imwrite(output_path, disp_norm)
    
    print(f"Disparity saved to: {output_path}")
    print(f"Disparity range: {disp_np.min():.3f} to {disp_np.max():.3f}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (will be slow)")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Load model
        print(f"Loading model from: {args.ckpt_dir}")
        model = FoundationStereo()
        checkpoint = torch.load(args.ckpt_dir, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
        # Load images
        print(f"Loading images:")
        print(f"  Left: {args.left_file}")
        print(f"  Right: {args.right_file}")
        
        image1 = load_image(args.left_file)
        image2 = load_image(args.right_file)
        
        print(f"Image shape: {image1.shape}")
        
        # Apply padding for network processing
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        
        # Run inference
        print("Running stereo matching...")
        with torch.no_grad():
            flow_pr = model(image1, image2, iters=args.valid_iters, test_mode=True)
        
        # Remove padding
        flow_pr = padder.unpad(flow_pr)
        
        # Save disparity
        output_file = os.path.join(args.out_dir, 'disparity.png')
        save_disparity(flow_pr, output_file)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
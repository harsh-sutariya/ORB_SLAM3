#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

def parse_args():
    parser = argparse.ArgumentParser(description='FoundationStereo Integration')
    parser.add_argument('--left_file', type=str, required=True, help='Left image path')
    parser.add_argument('--right_file', type=str, required=True, help='Right image path')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--out_dir', type=str, default='./outputs/', help='Output directory')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--valid_iters', type=int, default=32, help='Number of validation iterations')
    return parser.parse_args()

def load_image_pil(image_file):
    """Load and prepare image using PIL only"""
    img = Image.open(image_file).convert('RGB')
    img_array = np.array(img)
    return img_array

def save_disparity_simple(disparity, output_path):
    """Save disparity map using PIL only"""
    # Normalize to 0-255 range for visualization
    if disparity.max() > disparity.min():
        disp_norm = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disparity, dtype=np.uint8)
    
    # Save as PNG for visualization
    Image.fromarray(disp_norm).save(output_path)
    
    # Save raw disparity as 16-bit TIFF for precision
    disp_16bit = (disparity * 256).astype(np.uint16)
    Image.fromarray(disp_16bit).save(output_path.replace('.png', '.tiff'))
    
    print(f"Disparity saved to: {output_path}")
    print(f"Disparity range: {disparity.min():.3f} to {disparity.max():.3f}")

def create_basic_disparity(left_img, right_img):
    """Create a basic disparity map using simple block matching"""
    # Convert to grayscale
    if len(left_img.shape) == 3:
        left_gray = np.mean(left_img, axis=2).astype(np.uint8)
        right_gray = np.mean(right_img, axis=2).astype(np.uint8)
    else:
        left_gray = left_img.astype(np.uint8)
        right_gray = right_img.astype(np.uint8)
    
    height, width = left_gray.shape
    disparity = np.zeros_like(left_gray, dtype=np.float32)
    
    # Simple block matching parameters
    block_size = 15
    max_disparity = 64
    half_block = block_size // 2
    
    print("Computing basic block-matching disparity...")
    
    for y in range(half_block, height - half_block):
        if y % 50 == 0:
            print(f"Processing row {y}/{height}")
            
        for x in range(half_block + max_disparity, width - half_block):
            # Get template from left image
            template = left_gray[y-half_block:y+half_block+1, x-half_block:x+half_block+1]
            
            best_match_cost = float('inf')
            best_disparity = 0
            
            # Search in right image
            for d in range(max_disparity):
                if x - d - half_block < 0:
                    break
                    
                candidate = right_gray[y-half_block:y+half_block+1, 
                                     x-d-half_block:x-d+half_block+1]
                
                # Compute SAD (Sum of Absolute Differences)
                cost = np.sum(np.abs(template.astype(np.float32) - candidate.astype(np.float32)))
                
                if cost < best_match_cost:
                    best_match_cost = cost
                    best_disparity = d
            
            disparity[y, x] = best_disparity
    
    # Apply median filter to reduce noise
    from scipy.ndimage import median_filter
    disparity = median_filter(disparity, size=3)
    
    return disparity

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"FoundationStereo Integration Script")
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
    
    try:
        # Load images
        print("Loading stereo images...")
        left_img = load_image_pil(args.left_file)
        right_img = load_image_pil(args.right_file)
        print(f"Image dimensions: {left_img.shape}")
        
        # Try to import torch and run FoundationStereo
        try:
            import torch
            print("PyTorch available, attempting FoundationStereo...")
            
            # Try to import FoundationStereo components
            try:
                from omegaconf import OmegaConf
                from core.utils.utils import InputPadder
                from core.foundation_stereo import FoundationStereo
                
                # Check CUDA
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {device}")
                
                # Load model
                print(f"Loading FoundationStereo model from: {args.ckpt_dir}")
                model = FoundationStereo()
                checkpoint = torch.load(args.ckpt_dir, map_location=device)
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                model.eval()
                print("FoundationStereo model loaded successfully!")
                
                # Prepare images for model
                def prepare_image(img_array):
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                    return img_tensor[None].to(device)
                
                image1 = prepare_image(left_img)
                image2 = prepare_image(right_img)
                
                # Pad images
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                # Run inference
                print("Running FoundationStereo inference...")
                with torch.no_grad():
                    flow_pr = model(image1, image2, iters=args.valid_iters, test_mode=True)
                
                # Remove padding and convert to numpy
                flow_pr = padder.unpad(flow_pr)
                disparity = flow_pr.squeeze().cpu().numpy()
                
                print("FoundationStereo processing completed successfully!")
                
            except Exception as e:
                print(f"FoundationStereo model failed: {e}")
                print("Falling back to basic block matching...")
                disparity = create_basic_disparity(left_img, right_img)
                
        except ImportError:
            print("PyTorch not available, using basic block matching...")
            disparity = create_basic_disparity(left_img, right_img)
        
        # Save disparity
        output_file = os.path.join(args.out_dir, 'disparity.png')
        save_disparity_simple(disparity, output_file)
        
        print("Disparity computation completed!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
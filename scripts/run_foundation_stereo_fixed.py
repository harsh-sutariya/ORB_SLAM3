#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

def parse_args():
    parser = argparse.ArgumentParser(description='FoundationStereo Integration (Fixed)')
    parser.add_argument('--left_file', type=str, required=True, help='Left image path')
    parser.add_argument('--right_file', type=str, required=True, help='Right image path')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--out_dir', type=str, default='./outputs/', help='Output directory')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--valid_iters', type=int, default=32, help='Number of validation iterations')
    return parser.parse_args()

def create_euroc_intrinsics(out_dir):
    """Create camera intrinsics file for EuRoC dataset"""
    intrinsic_file = os.path.join(out_dir, 'K.txt')
    
    # EuRoC MAV dataset camera intrinsics (cam0)
    # These are approximate values for the EuRoC dataset
    fx = 458.654  # focal length x
    fy = 457.296  # focal length y  
    cx = 367.215  # principal point x
    cy = 248.375  # principal point y
    baseline = 0.110  # baseline between stereo cameras (meters)
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ])
    
    with open(intrinsic_file, 'w') as f:
        # Write intrinsic matrix as space-separated values
        f.write(' '.join(map(str, K.flatten())) + '\n')
        # Write baseline
        f.write(str(baseline) + '\n')
    
    return intrinsic_file

def save_disparity_opencv_format(disparity, output_path):
    """Save disparity map in OpenCV readable formats"""
    # Normalize to 0-255 range for visualization
    if disparity.max() > disparity.min():
        disp_norm = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disparity, dtype=np.uint8)
    
    # Save as PNG for visualization
    import cv2
    cv2.imwrite(output_path, disp_norm)
    
    # Save raw disparity as 16-bit TIFF for precision
    disp_16bit = (disparity * 256).astype(np.uint16)
    cv2.imwrite(output_path.replace('.png', '.tiff'), disp_16bit)
    
    print(f"Disparity saved to: {output_path}")
    print(f"Disparity range: {disparity.min():.3f} to {disparity.max():.3f}")

def run_foundation_stereo_model(left_img, right_img, ckpt_dir, intrinsic_file, args):
    """Run the actual FoundationStereo model"""
    try:
        import torch
        import imageio
        from omegaconf import OmegaConf
        from core.utils.utils import InputPadder
        from core.foundation_stereo import FoundationStereo
        
        print(f"Loading FoundationStereo model from: {ckpt_dir}")
        
        # Check if config file exists
        cfg_file = os.path.join(os.path.dirname(ckpt_dir), 'cfg.yaml')
        if not os.path.exists(cfg_file):
            print(f"Config file not found: {cfg_file}")
            return None
            
        # Load configuration
        cfg = OmegaConf.load(cfg_file)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
            
        # Update config with our arguments
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        args_updated = OmegaConf.create(cfg)
        
        # Initialize model
        model = FoundationStereo(args_updated)
        
        # Load checkpoint
        ckpt = torch.load(ckpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Checkpoint global_step: {ckpt.get('global_step', 'unknown')}, epoch: {ckpt.get('epoch', 'unknown')}")
        model.load_state_dict(ckpt['model'])
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"Using device: {device}")
        
        # Prepare images for model
        H, W = left_img.shape[:2]
        scale = args.scale
        
        if scale != 1.0:
            import cv2
            left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
            right_img = cv2.resize(right_img, fx=scale, fy=scale, dsize=None)
            H, W = left_img.shape[:2]
        
        # Convert to torch tensors
        img0 = torch.as_tensor(left_img).to(device).float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(right_img).to(device).float()[None].permute(0,3,1,2)
        
        # Pad images  
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        
        # Run inference
        print("Running FoundationStereo inference...")
        with torch.cuda.amp.autocast(True):
            with torch.no_grad():
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        
        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        
        print("FoundationStereo processing completed successfully!")
        return disp
        
    except Exception as e:
        print(f"FoundationStereo model failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_basic_disparity_improved(left_img, right_img):
    """Improved basic block matching disparity"""
    import cv2
    
    print("Running improved OpenCV stereo matching...")
    
    # Convert to grayscale
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    # Use OpenCV's StereoSGBM for better results
    # Parameters tuned for EuRoC dataset
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # Should be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5**2,  # Smoothness parameter 1
        P2=32 * 3 * 5**2,  # Smoothness parameter 2  
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Post-process to remove invalid disparities
    disparity[disparity <= 0] = 0
    disparity[disparity >= 64] = 0
    
    print(f"OpenCV SGBM disparity range: {disparity.min():.1f} to {disparity.max():.1f}")
    
    return disparity

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"FoundationStereo Integration Script (Fixed)")
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
        # Load images using imageio (like the original script)
        import imageio
        left_img = imageio.imread(args.left_file)
        right_img = imageio.imread(args.right_file)
        print(f"Image dimensions: {left_img.shape}")
        
        # Create camera intrinsics for EuRoC
        intrinsic_file = create_euroc_intrinsics(args.out_dir)
        print(f"Created camera intrinsics: {intrinsic_file}")
        
        # Try to run FoundationStereo model
        disparity = None
        if os.path.exists(args.ckpt_dir):
            try:
                import torch
                print("PyTorch available, attempting FoundationStereo...")
                disparity = run_foundation_stereo_model(left_img, right_img, args.ckpt_dir, intrinsic_file, args)
            except ImportError as e:
                print(f"Import error: {e}")
        else:
            print(f"Checkpoint not found: {args.ckpt_dir}")
        
        # Fallback to improved OpenCV stereo matching
        if disparity is None:
            print("Falling back to improved OpenCV stereo matching...")
            disparity = create_basic_disparity_improved(left_img, right_img)
        
        # Save disparity
        output_file = os.path.join(args.out_dir, 'disparity.png')
        save_disparity_opencv_format(disparity, output_file)
        
        print("Disparity computation completed!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
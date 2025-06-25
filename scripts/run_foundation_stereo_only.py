#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path to find the core modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

def parse_args():
    parser = argparse.ArgumentParser(description='FoundationStereo ONLY - No Fallbacks')
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

def run_foundation_stereo_model(left_img_path, right_img_path, ckpt_dir, intrinsic_file, args):
    """Run the FoundationStereo model - ONLY this, no fallbacks"""
    
    print("🚀 Loading FoundationStereo model...")
    print(f"📁 Checkpoint: {ckpt_dir}")
    print(f"🖼️  Left image: {left_img_path}")
    print(f"🖼️  Right image: {right_img_path}")
    
    try:
        # Import required modules
        import torch
        import imageio
        from omegaconf import OmegaConf
        from core.utils.utils import InputPadder
        from core.foundation_stereo import FoundationStereo
        
        print("✅ All dependencies imported successfully!")
        
        # Check if config file exists
        cfg_file = os.path.join(os.path.dirname(ckpt_dir), 'cfg.yaml')
        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
            
        print(f"📋 Loading config: {cfg_file}")
        
        # Load configuration
        cfg = OmegaConf.load(cfg_file)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
            
        # Update config with our arguments
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        args_updated = OmegaConf.create(cfg)
        
        print("🔧 Initializing FoundationStereo model...")
        
        # Initialize model
        model = FoundationStereo(args_updated)
        
        # Load checkpoint
        print(f"📦 Loading checkpoint: {ckpt_dir}")
        ckpt = torch.load(ckpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 Checkpoint info - global_step: {ckpt.get('global_step', 'unknown')}, epoch: {ckpt.get('epoch', 'unknown')}")
        model.load_state_dict(ckpt['model'])
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"🖥️  Using device: {device}")
        
        # Load and prepare images
        print("🖼️  Loading images...")
        left_img = imageio.imread(left_img_path)
        right_img = imageio.imread(right_img_path)
        
        print(f"📐 Original image dimensions: {left_img.shape}")
        
        # Apply scaling if needed
        H, W = left_img.shape[:2]
        scale = args.scale
        
        if scale != 1.0:
            import cv2
            left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
            right_img = cv2.resize(right_img, fx=scale, fy=scale, dsize=None)
            H, W = left_img.shape[:2]
            print(f"📐 Scaled image dimensions: {left_img.shape}")
        
        # Convert to torch tensors
        print("🔄 Converting images to tensors...")
        img0 = torch.as_tensor(left_img).to(device).float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(right_img).to(device).float()[None].permute(0,3,1,2)
        
        # Pad images for model requirements
        print("📏 Padding images...")
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        
        print(f"📐 Padded tensor shape: {img0.shape}")
        
        # Run FoundationStereo inference
        print(f"🧠 Running FoundationStereo inference (iterations: {args.valid_iters})...")
        
        with torch.cuda.amp.autocast(True):
            with torch.no_grad():
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        
        # Unpad and convert to numpy
        print("📤 Processing output...")
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        
        print(f"✅ FoundationStereo completed successfully!")
        print(f"📊 Disparity range: {disp.min():.3f} to {disp.max():.3f}")
        print(f"📐 Disparity shape: {disp.shape}")
        
        return disp
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the foundation_stereo conda environment")
        print("💡 Check if all required packages are installed")
        raise
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Check if the checkpoint and config files exist")
        raise
        
    except Exception as e:
        print(f"❌ FoundationStereo model failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def save_disparity_results(disparity, output_dir):
    """Save disparity map in multiple formats"""
    import cv2
    
    # Save raw disparity as numpy
    np.save(os.path.join(output_dir, 'disparity_raw.npy'), disparity)
    
    # Normalize to 0-255 range for visualization
    if disparity.max() > disparity.min():
        disp_norm = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disparity, dtype=np.uint8)
    
    # Save as PNG for visualization
    cv2.imwrite(os.path.join(output_dir, 'disparity.png'), disp_norm)
    
    # Save raw disparity as 16-bit TIFF for precision
    disp_16bit = (disparity * 256).astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, 'disparity.tiff'), disp_16bit)
    
    print(f"💾 Disparity saved in multiple formats:")
    print(f"   📁 Raw numpy: disparity_raw.npy")
    print(f"   🖼️  Visualization: disparity.png") 
    print(f"   📄 High precision: disparity.tiff")

def main():
    print("🎯 FoundationStereo ONLY Script - No Fallbacks!")
    print("=" * 60)
    
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"📋 Configuration:")
    print(f"   Left image: {args.left_file}")
    print(f"   Right image: {args.right_file}")
    print(f"   Checkpoint: {args.ckpt_dir}")
    print(f"   Output dir: {args.out_dir}")
    print(f"   Scale: {args.scale}")
    print(f"   Iterations: {args.valid_iters}")
    print()
    
    # Check if files exist
    if not os.path.exists(args.left_file):
        raise FileNotFoundError(f"Left image not found: {args.left_file}")
        
    if not os.path.exists(args.right_file):
        raise FileNotFoundError(f"Right image not found: {args.right_file}")
    
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_dir}")
    
    try:
        # Create camera intrinsics for EuRoC
        intrinsic_file = create_euroc_intrinsics(args.out_dir)
        print(f"📷 Created camera intrinsics: {intrinsic_file}")
        print()
        
        # Run FoundationStereo model - NO FALLBACKS!
        disparity = run_foundation_stereo_model(
            args.left_file, 
            args.right_file, 
            args.ckpt_dir, 
            intrinsic_file, 
            args
        )
        
        print()
        print("💾 Saving results...")
        save_disparity_results(disparity, args.out_dir)
        
        print()
        print("🎉 FoundationStereo processing completed successfully!")
        print(f"📊 Final disparity stats:")
        print(f"   Shape: {disparity.shape}")
        print(f"   Range: {disparity.min():.3f} to {disparity.max():.3f}")
        print(f"   Mean: {disparity.mean():.3f}")
        print(f"   Std: {disparity.std():.3f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("🔧 This script only runs FoundationStereo - no fallbacks!")
        print("💡 Please ensure:")
        print("   - foundation_stereo conda environment is active")
        print("   - All dependencies are properly installed")
        print("   - Checkpoint and config files exist")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
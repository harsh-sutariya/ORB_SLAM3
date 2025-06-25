#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import numpy as np
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='FoundationStereo via subprocess isolation')
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

def run_foundation_stereo_subprocess(left_img_path, right_img_path, ckpt_dir, intrinsic_file, args):
    """Run FoundationStereo via subprocess to avoid import conflicts"""
    
    print("🚀 Running FoundationStereo via subprocess isolation...")
    
    # FoundationStereo directory
    fs_dir = "/home/lunar/FoundationStereo"
    
    if not os.path.exists(fs_dir):
        raise FileNotFoundError(f"FoundationStereo directory not found: {fs_dir}")
    
    # Map ckpt_dir to FoundationStereo directory
    if ckpt_dir.startswith('./pretrained_models/'):
        fs_ckpt_dir = os.path.join(fs_dir, ckpt_dir[2:])  # Remove './'
    else:
        fs_ckpt_dir = ckpt_dir
    
    if not os.path.exists(fs_ckpt_dir):
        print(f"⚠️  Checkpoint not found at {fs_ckpt_dir}, trying original location...")
        fs_ckpt_dir = ckpt_dir
    
    print(f"📁 Using checkpoint: {fs_ckpt_dir}")
    print(f"📁 Output directory: {args.out_dir}")
    print(f"🖼️  Left image: {left_img_path}")
    print(f"🖼️  Right image: {right_img_path}")
    
    # Create the command to run FoundationStereo
    cmd = [
        "python", 
                 "scripts/run_demo_fixed.py",
                 "--left_file", os.path.abspath(left_img_path),
         "--right_file", os.path.abspath(right_img_path),
                 "--intrinsic_file", os.path.abspath(intrinsic_file),
        "--ckpt_dir", fs_ckpt_dir,
                 "--out_dir", os.path.abspath(args.out_dir),
        "--valid_iters", str(args.valid_iters)
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"📂 Working directory: {fs_dir}")
    
    # Run the command in FoundationStereo directory
    try:
        result = subprocess.run(
            cmd,
            cwd=fs_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("📤 FoundationStereo subprocess output:")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✅ FoundationStereo completed successfully!")
            
            # Check if depth output was generated
            depth_file = os.path.join(args.out_dir, 'depth_meter.npy')
            if os.path.exists(depth_file):
                print(f"📊 Loading depth file: {depth_file}")
                depth = np.load(depth_file)
                
                # Convert depth to disparity: disparity = (fx * baseline) / depth
                fx = 458.654
                baseline = 0.110
                
                disparity = (fx * baseline) / depth
                disparity[depth <= 0] = 0
                disparity[disparity < 0] = 0
                disparity[disparity > 100] = 0  # Reasonable disparity limit
                
                print(f"📊 Disparity stats:")
                print(f"   Shape: {disparity.shape}")
                print(f"   Range: {disparity.min():.3f} to {disparity.max():.3f}")
                print(f"   Mean: {disparity.mean():.3f}")
                
                return disparity
            else:
                print(f"⚠️  Depth file not found: {depth_file}")
                return None
        else:
            print(f"❌ FoundationStereo failed with return code: {result.returncode}")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏰ FoundationStereo subprocess timed out!")
        return None
    except Exception as e:
        print(f"❌ Subprocess error: {e}")
        return None

def save_disparity_results(disparity, output_dir):
    """Save disparity map in multiple formats using basic methods"""
    
    # Save raw disparity as numpy
    np.save(os.path.join(output_dir, 'disparity_raw.npy'), disparity)
    
    # Create a simple subprocess script for OpenCV operations to avoid import issues
    opencv_script = f'''
import cv2
import numpy as np

# Load disparity
disparity = np.load("{output_dir}/disparity_raw.npy")

# Normalize to 0-255 range for visualization
if disparity.max() > disparity.min():
    disp_norm = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
else:
    disp_norm = np.zeros_like(disparity, dtype=np.uint8)

# Save as PNG for visualization
cv2.imwrite("{output_dir}/disparity.png", disp_norm)

# Save raw disparity as 16-bit TIFF for precision
disp_16bit = (disparity * 256).astype(np.uint16)
cv2.imwrite("{output_dir}/disparity.tiff", disp_16bit)

print(f"💾 Disparity saved:")
print(f"   📁 Raw numpy: disparity_raw.npy")
print(f"   🖼️  Visualization: disparity.png") 
print(f"   📄 High precision: disparity.tiff")
print(f"   📊 Range: {{disparity.min():.3f}} to {{disparity.max():.3f}}")
'''
    
    # Write and execute the OpenCV script
    script_path = os.path.join(output_dir, 'save_script.py')
    with open(script_path, 'w') as f:
        f.write(opencv_script)
    
    try:
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"⚠️  OpenCV save script failed: {result.stderr}")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def main():
    print("🎯 FoundationStereo via Subprocess Isolation!")
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
    
    try:
        # Create camera intrinsics for EuRoC
        intrinsic_file = create_euroc_intrinsics(args.out_dir)
        print(f"📷 Created camera intrinsics: {intrinsic_file}")
        print()
        
        # Run FoundationStereo via subprocess
        disparity = run_foundation_stereo_subprocess(
            args.left_file, 
            args.right_file, 
            args.ckpt_dir, 
            intrinsic_file, 
            args
        )
        
        if disparity is not None:
            print()
            print("💾 Saving results...")
            save_disparity_results(disparity, args.out_dir)
            
            print()
            print("🎉 FoundationStereo processing completed successfully!")
            return 0
        else:
            print("❌ FoundationStereo processing failed!")
            return 1
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
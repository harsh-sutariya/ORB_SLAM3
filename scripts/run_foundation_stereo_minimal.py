#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='FoundationStereo Integration (Minimal)')
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

def run_foundation_stereo_with_isolation(left_img_path, right_img_path, ckpt_dir, intrinsic_file, args):
    """Try to run FoundationStereo in isolated environment"""
    try:
        # Change to FoundationStereo directory to avoid path issues
        fs_dir = "/home/lunar/FoundationStereo"
        if os.path.exists(fs_dir):
            print(f"Attempting to run FoundationStereo from: {fs_dir}")
            
            # Construct command to run in FoundationStereo directory
            cmd = f"cd {fs_dir} && python scripts/run_demo.py"
            cmd += f" --left_file {left_img_path}"
            cmd += f" --right_file {right_img_path}"
            cmd += f" --intrinsic_file {intrinsic_file}"
            cmd += f" --ckpt_dir {ckpt_dir.replace('./pretrained_models/', f'{fs_dir}/pretrained_models/')}"
            cmd += f" --out_dir {args.out_dir}"
            cmd += f" --valid_iters {args.valid_iters}"
            
            print(f"Running: {cmd}")
            result = os.system(cmd)
            
            if result == 0:
                # Check if output was generated
                depth_file = os.path.join(args.out_dir, 'depth_meter.npy')
                if os.path.exists(depth_file):
                    depth = np.load(depth_file)
                    
                    # Convert depth to disparity: disparity = (fx * baseline) / depth
                    K = np.array([458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1]).reshape(3, 3)
                    baseline = 0.110
                    
                    disparity = (K[0,0] * baseline) / depth
                    disparity[depth <= 0] = 0
                    disparity[disparity < 0] = 0
                    disparity[disparity > 64] = 0
                    
                    print("FoundationStereo completed successfully!")
                    return disparity
            
        return None
        
    except Exception as e:
        print(f"FoundationStereo isolated run failed: {e}")
        return None

def create_enhanced_block_matching(left_img_path, right_img_path):
    """Enhanced block matching using system OpenCV to avoid conflicts"""
    import subprocess
    import tempfile
    
    print("Running enhanced OpenCV stereo matching via subprocess...")
    
    # Create a standalone OpenCV script to avoid import conflicts
    opencv_script = f"""
import cv2
import numpy as np
import sys

def enhanced_stereo_matching(left_path, right_path, output_path):
    # Load images
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if left is None or right is None:
        print("Failed to load images")
        return False
    
    # Enhanced StereoSGBM parameters for EuRoC dataset
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=80,  # Must be divisible by 16
        blockSize=3,
        P1=8 * 3 * 3**2,
        P2=32 * 3 * 3**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0
    
    # Clean up disparity
    disparity[disparity <= 0] = 0
    disparity[disparity >= 80] = 0
    
    # Save as 16-bit for precision
    disp_16bit = (disparity * 256).astype(np.uint16)
    cv2.imwrite(output_path, disp_16bit)
    
    print(f"Enhanced OpenCV disparity range: {{disparity.min():.1f}} to {{disparity.max():.1f}}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: script.py left_img right_img output_path")
        sys.exit(1)
    
    success = enhanced_stereo_matching(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if success else 1)
"""
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(opencv_script)
        temp_script = f.name
    
    try:
        # Output file
        temp_output = os.path.join(args.out_dir, 'disparity_temp.png')
        
        # Run the script using system python (not conda environment)
        cmd = ['/usr/bin/python3', temp_script, left_img_path, right_img_path, temp_output]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(temp_output):
            print("Enhanced OpenCV stereo matching completed successfully!")
            
            # Load the result using system OpenCV
            import subprocess
            load_script = f"""
import cv2
import numpy as np
disparity = cv2.imread('{temp_output}', cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
np.save('{args.out_dir}/disparity_opencv.npy', disparity)
print(f"Saved disparity shape: {{disparity.shape}}, range: {{disparity.min():.1f}} to {{disparity.max():.1f}}")
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(load_script)
                load_temp = f.name
            
            subprocess.run(['/usr/bin/python3', load_temp], check=True)
            
            # Load the numpy result
            disparity = np.load(f'{args.out_dir}/disparity_opencv.npy')
            os.unlink(load_temp)
            os.unlink(temp_output)
            
            return disparity
        else:
            print(f"OpenCV subprocess failed: {result.stderr}")
            return None
            
    finally:
        if os.path.exists(temp_script):
            os.unlink(temp_script)

def save_disparity_opencv_format(disparity, output_path):
    """Save disparity map in OpenCV readable formats"""
    # Normalize to 0-255 range for visualization
    if disparity.max() > disparity.min():
        disp_norm = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disparity, dtype=np.uint8)
    
    # Save as PNG for visualization (using subprocess to avoid import conflicts)
    import subprocess
    import tempfile
    
    save_script = f"""
import cv2
import numpy as np
disparity = np.load('{args.out_dir}/disparity_temp.npy')
disp_norm = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8) if disparity.max() > disparity.min() else np.zeros_like(disparity, dtype=np.uint8)
cv2.imwrite('{output_path}', disp_norm)
disp_16bit = (disparity * 256).astype(np.uint16)
cv2.imwrite('{output_path.replace(".png", ".tiff")}', disp_16bit)
print(f"Disparity saved to: {output_path}")
print(f"Disparity range: {{disparity.min():.3f}} to {{disparity.max():.3f}}")
"""
    
    # Save disparity data temporarily
    np.save(f'{args.out_dir}/disparity_temp.npy', disparity)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(save_script)
        temp_script = f.name
    
    try:
        subprocess.run(['/usr/bin/python3', temp_script], check=True)
        os.unlink(f'{args.out_dir}/disparity_temp.npy')
    finally:
        if os.path.exists(temp_script):
            os.unlink(temp_script)

def main():
    global args
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"FoundationStereo Integration Script (Minimal)")
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
        # Create camera intrinsics for EuRoC
        intrinsic_file = create_euroc_intrinsics(args.out_dir)
        print(f"Created camera intrinsics: {intrinsic_file}")
        
        # Try to run FoundationStereo in isolation
        disparity = None
        if os.path.exists(args.ckpt_dir) or os.path.exists("/home/lunar/FoundationStereo"):
            print("Attempting FoundationStereo in isolated environment...")
            disparity = run_foundation_stereo_with_isolation(args.left_file, args.right_file, args.ckpt_dir, intrinsic_file, args)
        
        # Fallback to enhanced OpenCV stereo matching
        if disparity is None:
            print("Falling back to enhanced OpenCV stereo matching...")
            disparity = create_enhanced_block_matching(args.left_file, args.right_file)
        
        if disparity is not None:
            # Save disparity
            output_file = os.path.join(args.out_dir, 'disparity.png')
            save_disparity_opencv_format(disparity, output_file)
            print("Disparity computation completed!")
        else:
            print("All disparity computation methods failed!")
            return 1
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
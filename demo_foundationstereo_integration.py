#!/usr/bin/env python3
"""
Demo script showing FoundationStereo integration with ORB-SLAM3

This script demonstrates:
1. How to use the GetStereoDisparity function
2. Performance comparison between traditional and FoundationStereo
3. Quality assessment of disparity maps
"""

import os
import cv2
import numpy as np
import time
import subprocess
import argparse
from pathlib import Path

def load_euroc_test_images():
    """Load test stereo pair from EuRoC dataset"""
    
    # Check if we have the test images from our previous tests
    test_dir = "test_outputs"
    left_img_path = None
    right_img_path = None
    
    # Look for existing test images
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if "left.png" in file:
                left_img_path = os.path.join(root, file)
            elif "right.png" in file:
                right_img_path = os.path.join(root, file)
                
        if left_img_path and right_img_path:
            break
    
    if left_img_path and right_img_path:
        print(f"Using existing test images:")
        print(f"Left: {left_img_path}")
        print(f"Right: {right_img_path}")
        return cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE), cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    # If no test images found, create dummy stereo pair
    print("No existing test images found. Creating dummy stereo pair...")
    height, width = 480, 752
    left_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Add some matching features (shifted pattern)
    pattern = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(pattern, (20, 20), (80, 80), 255, -1)
    
    # Place pattern in both images with horizontal disparity
    y_start, x_start = 200, 300
    left_img[y_start:y_start+100, x_start:x_start+100] = pattern
    right_img[y_start:y_start+100, x_start-20:x_start+80] = pattern  # 20 pixel disparity
    
    return left_img, right_img

def test_traditional_stereo(left_img, right_img):
    """Test traditional OpenCV stereo matching"""
    print("\n=== Testing Traditional Stereo Matching ===")
    start_time = time.time()
    
    # Create stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    
    # Compute disparity
    disparity = stereo.compute(left_img, right_img)
    
    # Convert to float and normalize
    disparity = disparity.astype(np.float32) / 16.0
    
    end_time = time.time()
    
    # Statistics
    valid_mask = disparity > 0
    valid_pixels = np.sum(valid_mask)
    mean_disparity = np.mean(disparity[valid_mask]) if valid_pixels > 0 else 0
    
    print(f"Traditional stereo - Time: {end_time - start_time:.3f}s")
    print(f"Valid pixels: {valid_pixels}/{disparity.size} ({100*valid_pixels/disparity.size:.1f}%)")
    print(f"Mean disparity: {mean_disparity:.2f}")
    
    return disparity, end_time - start_time

def test_foundationstereo(left_img, right_img, output_dir="./demo_output"):
    """Test FoundationStereo disparity computation"""
    print("\n=== Testing FoundationStereo ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images
    left_path = os.path.join(output_dir, "left.png")
    right_path = os.path.join(output_dir, "right.png")
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)
    
    print(f"Saved test images to {output_dir}")
    
    start_time = time.time()
    
    # Call FoundationStereo script
    cmd = [
        "python", "scripts/run_foundation_stereo_isolated.py",
        "--left_file", left_path,
        "--right_file", right_path,
        "--ckpt_dir", "./pretrained_models/23-51-11/model_best_bp2.pth",
        "--out_dir", output_dir,
        "--valid_iters", "32"
    ]
    
    try:
        print("Running FoundationStereo...")
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("FoundationStereo completed successfully!")
            
            # Load disparity result
            disparity_path = os.path.join(output_dir, "disparity.tiff")
            if os.path.exists(disparity_path):
                disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)
                end_time = time.time()
                
                # Statistics
                valid_mask = disparity > 0
                valid_pixels = np.sum(valid_mask)
                mean_disparity = np.mean(disparity[valid_mask]) if valid_pixels > 0 else 0
                
                print(f"FoundationStereo - Time: {end_time - start_time:.3f}s")
                print(f"Valid pixels: {valid_pixels}/{disparity.size} ({100*valid_pixels/disparity.size:.1f}%)")
                print(f"Mean disparity: {mean_disparity:.2f}")
                print(f"Disparity range: {np.min(disparity[valid_mask]):.2f} - {np.max(disparity[valid_mask]):.2f}")
                
                return disparity, end_time - start_time
            else:
                print(f"Error: Disparity file not found at {disparity_path}")
                return None, time.time() - start_time
                
        else:
            print(f"FoundationStereo failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return None, time.time() - start_time
            
    except subprocess.TimeoutExpired:
        print("FoundationStereo timed out (60s limit)")
        return None, 60.0
    except Exception as e:
        print(f"Error running FoundationStereo: {e}")
        return None, time.time() - start_time

def visualize_results(left_img, traditional_disp, foundation_disp, output_dir):
    """Create visualization comparing traditional vs FoundationStereo results"""
    print("\n=== Creating Visualizations ===")
    
    # Normalize disparities for visualization
    def normalize_disparity(disp):
        if disp is None:
            return np.zeros_like(left_img)
        valid_mask = disp > 0
        if np.sum(valid_mask) == 0:
            return np.zeros_like(left_img)
        disp_norm = np.zeros_like(disp)
        disp_norm[valid_mask] = 255 * (disp[valid_mask] - np.min(disp[valid_mask])) / (np.max(disp[valid_mask]) - np.min(disp[valid_mask]))
        return disp_norm.astype(np.uint8)
    
    trad_vis = normalize_disparity(traditional_disp)
    found_vis = normalize_disparity(foundation_disp) if foundation_disp is not None else np.zeros_like(left_img)
    
    # Create comparison image
    h, w = left_img.shape
    comparison = np.zeros((h*2, w*2), dtype=np.uint8)
    
    # Top row: original images
    comparison[:h, :w] = left_img
    comparison[:h, w:] = cv2.imread(os.path.join(output_dir, "right.png"), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(output_dir, "right.png")) else left_img
    
    # Bottom row: disparity maps
    comparison[h:, :w] = trad_vis
    comparison[h:, w:] = found_vis
    
    # Add labels
    comparison_color = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_color, "Left Image", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_color, "Right Image", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_color, "Traditional Stereo", (10, h + 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison_color, "FoundationStereo", (w + 10, h + 30), font, 1, (255, 255, 255), 2)
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "stereo_comparison.png")
    cv2.imwrite(comparison_path, comparison_color)
    print(f"Saved comparison visualization to {comparison_path}")
    
    return comparison_path

def demonstrate_orbslam_integration():
    """Show how to integrate FoundationStereo with ORB-SLAM3"""
    print("\n=== ORB-SLAM3 Integration Example ===")
    
    integration_code = '''
    // In your ORB-SLAM3 application:
    
    // 1. Create a Frame instance
    ORB_SLAM3::Frame frame;
    
    // 2. Compute FoundationStereo disparity
    cv::Mat disparity = frame.GetStereoDisparity(imLeft, imRight, "./output");
    
    // 3. Use disparity for enhanced depth estimation
    if (!disparity.empty()) {
        // Disparity is now available for:
        // - Improving stereo matching accuracy
        // - Filling gaps in traditional stereo
        // - Providing dense depth maps
        // - Point cloud generation
    }
    
    // 4. Run normal ORB-SLAM3 tracking
    SLAM.TrackStereo(imLeft, imRight, timestamp);
    '''
    
    print("C++ Integration Code:")
    print(integration_code)
    
    print("\nBuilding the FoundationStereo example:")
    print("cd /home/lunar/ORB_SLAM3/build")
    print("make stereo_euroc_foundationstereo")
    
    print("\nRunning with EuRoC dataset:")
    print("./stereo_euroc_foundationstereo Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml \\")
    print("    /path/to/euroc/MH_01_easy Examples/Stereo/EuRoC_TimeStamps/MH01.txt \\")
    print("    trajectory_foundationstereo")

def main():
    parser = argparse.ArgumentParser(description="Demo FoundationStereo integration with ORB-SLAM3")
    parser.add_argument("--output_dir", default="./demo_output", help="Output directory for results")
    parser.add_argument("--skip_foundationstereo", action="store_true", help="Skip FoundationStereo test (faster)")
    args = parser.parse_args()
    
    print("FoundationStereo + ORB-SLAM3 Integration Demo")
    print("=" * 50)
    
    # Load test images
    print("Loading test stereo pair...")
    left_img, right_img = load_euroc_test_images()
    print(f"Image size: {left_img.shape}")
    
    # Test traditional stereo
    traditional_disp, trad_time = test_traditional_stereo(left_img, right_img)
    
    # Test FoundationStereo
    foundation_disp, found_time = None, 0
    if not args.skip_foundationstereo:
        foundation_disp, found_time = test_foundationstereo(left_img, right_img, args.output_dir)
    else:
        print("\nSkipping FoundationStereo test (--skip_foundationstereo flag)")
    
    # Create visualizations
    comparison_path = visualize_results(left_img, traditional_disp, foundation_disp, args.output_dir)
    
    # Performance summary
    print("\n=== Performance Summary ===")
    print(f"Traditional Stereo: {trad_time:.3f}s")
    if not args.skip_foundationstereo:
        print(f"FoundationStereo: {found_time:.3f}s")
        if found_time > 0:
            print(f"Speed ratio: {found_time/trad_time:.1f}x slower")
    
    # Integration example
    demonstrate_orbslam_integration()
    
    print(f"\nDemo completed! Check results in: {args.output_dir}")
    print(f"Comparison image: {comparison_path}")

if __name__ == "__main__":
    main() 
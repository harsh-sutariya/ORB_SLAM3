#!/usr/bin/env python3

"""
Streamlined ROS Bag to Ground Truth Extraction
==============================================

This script consolidates the entire ground truth processing pipeline:

INPUTS:
  - ROS bag file with topics: /robot/pose, /camera/left/image_raw, /camera/right/image_raw, /imu/data

PROCESSING:
  1. Extract and normalize robot poses to start from origin (0,0,0)
  2. Normalize timestamps to start from 0.000000000 seconds  
  3. Extract camera images with synchronized timestamps
  4. Extract IMU measurements
  5. Generate sensor configuration files

OUTPUTS:
  📁 mav0/                           # EuRoC dataset format
    ├── cam0/data.csv               # Left camera timestamps
    ├── cam1/data.csv               # Right camera timestamps  
    ├── imu0/data.csv               # IMU measurements
    ├── state_groundtruth_estimate0/
    │   ├── data.csv                # Ground truth (EuRoC CSV format)
    │   ├── groundtruth_normalized.txt  # TUM format, ready for evaluation
    │   ├── groundtruth.txt         # TUM format with original timestamps
    │   └── transformation_info.yaml # Normalization metadata
    └── sensor.yaml files
  📄 transformation_info.yaml        # Conversion metadata & validation stats

ELIMINATED REDUNDANCY:
  ❌ No longer need: evaluate_stereo_comparison.py convert_euroc_to_tum()
  ❌ No longer need: fix_timestamp_alignment.py normalize_ground_truth_timestamps()
  ✅ Single-step processing from ROS bag to evaluation-ready formats

USAGE:
  python3 extract_bag_to_mav0.py <bag_file> <output_directory>
"""

import rosbag
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import yaml
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
import argparse
from tqdm import tqdm
import pandas as pd
import tf.transformations as tf_trans

def create_mav0_structure(output_dir):
    """Create the mav0 directory structure"""
    mav0_dir = os.path.join(output_dir, "mav0")
    
    # Create directories
    cam0_dir = os.path.join(mav0_dir, "cam0", "data")
    cam1_dir = os.path.join(mav0_dir, "cam1", "data")
    imu0_dir = os.path.join(mav0_dir, "imu0")
    gt_dir = os.path.join(mav0_dir, "state_groundtruth_estimate0")
    
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(imu0_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    return mav0_dir, cam0_dir, cam1_dir, imu0_dir, gt_dir

def timestamp_to_nanoseconds(stamp):
    """Convert ROS timestamp to nanoseconds"""
    return int(stamp.secs * 1e9 + stamp.nsecs)

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    return tf_trans.quaternion_matrix([q[1], q[2], q[3], q[0]])  # tf uses [x,y,z,w] format

def pose_to_transformation_matrix(position, orientation):
    """Convert position and quaternion to 4x4 transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = tf_trans.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])[:3, :3]
    T[:3, 3] = [position.x, position.y, position.z]
    return T

def transform_pose(pose, T_inv):
    """Transform a pose using inverse transformation matrix"""
    # Convert pose to homogeneous coordinates
    pose_matrix = pose_to_transformation_matrix(pose.position, pose.orientation)
    
    # Apply transformation
    transformed_matrix = T_inv @ pose_matrix
    
    # Extract position and orientation
    transformed_position = transformed_matrix[:3, 3]
    transformed_quaternion = tf_trans.quaternion_from_matrix(transformed_matrix)
    
    return transformed_position, transformed_quaternion

def extract_bag_data(bag_path, output_dir):
    """Extract data from ROS bag to mav0 format"""
    
    print(f"Processing bag: {bag_path}")
    print(f"Output directory: {output_dir}")
    
    # Create directory structure
    mav0_dir, cam0_dir, cam1_dir, imu0_dir, gt_dir = create_mav0_structure(output_dir)
    
    # Initialize bridge for image conversion
    bridge = CvBridge()
    
    # Data storage
    cam0_data = []  # left camera
    cam1_data = []  # right camera
    imu_data = []
    gt_data = []    # ground truth poses (CSV format)
    gt_tum_data = []  # ground truth poses (TUM format)
    camera_info_left = None
    camera_info_right = None
    tf_transforms = []  # tf transformations
    first_gt_pose = None  # first ground truth pose for normalization
    first_timestamp = None  # first timestamp for normalization
    T_inv_normalization = None  # transformation matrix to normalize to origin
    
    print("Opening bag file...")
    bag = rosbag.Bag(bag_path, 'r')
    
    # Get total message count for progress bar
    total_messages = bag.get_message_count()
    
    print("Extracting data...")
    with tqdm(total=total_messages, desc="Processing messages") as pbar:
        for topic, msg, t in bag.read_messages():
            pbar.update(1)
            
            if topic == '/camera/left/image_raw':
                # Convert ROS image to OpenCV
                cv_image = bridge.imgmsg_to_cv2(msg, "mono8")
                timestamp_ns = timestamp_to_nanoseconds(msg.header.stamp)
                
                # Save image
                image_filename = f"{timestamp_ns}.png"
                image_path = os.path.join(cam0_dir, image_filename)
                cv2.imwrite(image_path, cv_image)
                
                # Store timestamp data
                cam0_data.append([timestamp_ns, image_filename])
                
            elif topic == '/camera/right/image_raw':
                # Convert ROS image to OpenCV
                cv_image = bridge.imgmsg_to_cv2(msg, "mono8")
                timestamp_ns = timestamp_to_nanoseconds(msg.header.stamp)
                
                # Save image
                image_filename = f"{timestamp_ns}.png"
                image_path = os.path.join(cam1_dir, image_filename)
                cv2.imwrite(image_path, cv_image)
                
                # Store timestamp data
                cam1_data.append([timestamp_ns, image_filename])
                
            elif topic == '/camera/left/camera_info':
                if camera_info_left is None:
                    camera_info_left = msg
                    
            elif topic == '/camera/right/camera_info':
                if camera_info_right is None:
                    camera_info_right = msg
                    
            elif topic == '/imu/data':
                timestamp_ns = timestamp_to_nanoseconds(msg.header.stamp)
                
                # Extract IMU data: [timestamp, omega_x, omega_y, omega_z, alpha_x, alpha_y, alpha_z]
                imu_data.append([
                    timestamp_ns,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])
                
            elif topic == '/robot/pose':
                timestamp_ns = timestamp_to_nanoseconds(msg.header.stamp)
                timestamp_s = timestamp_ns / 1e9  # Convert to seconds for TUM format
                
                # Store first pose and timestamp for normalization
                if first_gt_pose is None:
                    first_gt_pose = msg.pose
                    first_timestamp = timestamp_s
                    # Create inverse transformation matrix to normalize to origin
                    T_first = pose_to_transformation_matrix(first_gt_pose.position, first_gt_pose.orientation)
                    T_inv_normalization = np.linalg.inv(T_first)
                    print(f"🎯 Ground truth normalization references:")
                    print(f"  First pose: ({first_gt_pose.position.x:.3f}, {first_gt_pose.position.y:.3f}, {first_gt_pose.position.z:.3f})")
                    print(f"  First timestamp: {first_timestamp:.9f}")
                
                # Transform pose to start from origin
                if T_inv_normalization is not None:
                    transformed_pos, transformed_quat = transform_pose(msg.pose, T_inv_normalization)
                    
                    # Normalize timestamp to start from 0
                    normalized_timestamp = timestamp_s - first_timestamp
                    
                    # Store CSV format data (EuRoC compatibility): [timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z]
                    gt_data.append([
                        timestamp_ns,
                        transformed_pos[0],
                        transformed_pos[1], 
                        transformed_pos[2],
                        transformed_quat[3],  # w
                        transformed_quat[0],  # x
                        transformed_quat[1],  # y
                        transformed_quat[2]   # z
                    ])
                    
                    # Store TUM format data: [timestamp_s, p_x, p_y, p_z, q_x, q_y, q_z, q_w]
                    gt_tum_data.append([
                        normalized_timestamp,  # Normalized timestamp in seconds
                        transformed_pos[0],    # x
                        transformed_pos[1],    # y
                        transformed_pos[2],    # z
                        transformed_quat[0],   # qx
                        transformed_quat[1],   # qy
                        transformed_quat[2],   # qz
                        transformed_quat[3]    # qw
                    ])
                else:
                    # Fallback to original pose if transformation failed
                    normalized_timestamp = timestamp_s - (first_timestamp if first_timestamp else 0)
                    
                    gt_data.append([
                        timestamp_ns,
                        msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        msg.pose.orientation.w,
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z
                    ])
                    
                    gt_tum_data.append([
                        normalized_timestamp,
                        msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w
                    ])
                    
            elif topic == '/tf_static':
                # Extract tf transformations for reference
                for transform in msg.transforms:
                    tf_transforms.append({
                        'parent_frame': transform.header.frame_id,
                        'child_frame': transform.child_frame_id,
                        'translation': [transform.transform.translation.x, 
                                      transform.transform.translation.y, 
                                      transform.transform.translation.z],
                        'rotation': [transform.transform.rotation.x, 
                                   transform.transform.rotation.y, 
                                   transform.transform.rotation.z, 
                                   transform.transform.rotation.w]
                    })
                    print(f"Found tf transform: {transform.header.frame_id} -> {transform.child_frame_id}")
    
    bag.close()
    
    # Print transformation summary
    if tf_transforms:
        print(f"\nFound {len(tf_transforms)} TF transformations:")
        for tf in tf_transforms:
            print(f"  {tf['parent_frame']} -> {tf['child_frame']}")
            print(f"    Translation: {tf['translation']}")
            print(f"    Rotation: {tf['rotation']}")
    
    if first_gt_pose is not None:
        print(f"\n✅ Ground truth normalization applied:")
        print(f"  Original first pose: ({first_gt_pose.position.x:.3f}, {first_gt_pose.position.y:.3f}, {first_gt_pose.position.z:.3f})")
        print(f"  Normalized to start from origin (0, 0, 0)")
        print(f"  Original first timestamp: {first_timestamp:.9f}")
        print(f"  Normalized timestamps to start from 0.000000000")
    
    print("💾 Saving data files...")
    
    # Save camera data CSV files
    if cam0_data:
        cam0_df = pd.DataFrame(cam0_data, columns=['#timestamp [ns]', 'filename'])
        cam0_df.to_csv(os.path.join(mav0_dir, "cam0", "data.csv"), index=False)
        print(f"📷 Saved {len(cam0_data)} left camera images")
    
    if cam1_data:
        cam1_df = pd.DataFrame(cam1_data, columns=['#timestamp [ns]', 'filename'])
        cam1_df.to_csv(os.path.join(mav0_dir, "cam1", "data.csv"), index=False)
        print(f"📷 Saved {len(cam1_data)} right camera images")
    
    # Save IMU data
    if imu_data:
        imu_df = pd.DataFrame(imu_data, columns=[
            '#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]'
        ])
        imu_df.to_csv(os.path.join(imu0_dir, "data.csv"), index=False)
        print(f"📐 Saved {len(imu_data)} IMU measurements")
    
    # Save ground truth data in multiple formats
    if gt_data:
        # 1. Save EuRoC CSV format (for compatibility)
        gt_df = pd.DataFrame(gt_data, columns=[
            '#timestamp [ns]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]',
            'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
        ])
        gt_df.to_csv(os.path.join(gt_dir, "data.csv"), index=False)
        print(f"📊 Saved {len(gt_data)} ground truth poses (EuRoC CSV format)")
        
        # 2. Save TUM format for direct evaluation (normalized)
        gt_tum_path = os.path.join(gt_dir, "groundtruth_normalized.txt")
        with open(gt_tum_path, 'w') as f:
            for pose in gt_tum_data:
                f.write(f"{pose[0]:.9f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f} {pose[7]:.6f}\n")
        print(f"🎯 Saved {len(gt_tum_data)} ground truth poses (TUM format, normalized): {gt_tum_path}")
        
        # 3. Also save original TUM format (with Unix timestamps) for reference
        gt_tum_orig_path = os.path.join(gt_dir, "groundtruth.txt")
        with open(gt_tum_orig_path, 'w') as f:
            for i, pose in enumerate(gt_tum_data):
                # Add back the original timestamp for reference
                original_timestamp = pose[0] + first_timestamp if first_timestamp else pose[0]
                f.write(f"{original_timestamp:.9f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f} {pose[7]:.6f}\n")
        print(f"📄 Saved ground truth with original timestamps (TUM format): {gt_tum_orig_path}")
    
    # Save transformation information
    if tf_transforms or first_gt_pose is not None:
        transform_info = {
            'tf_transforms': tf_transforms,
            'original_first_pose': {
                'position': [first_gt_pose.position.x, first_gt_pose.position.y, first_gt_pose.position.z],
                'orientation': [first_gt_pose.orientation.x, first_gt_pose.orientation.y, 
                              first_gt_pose.orientation.z, first_gt_pose.orientation.w]
            } if first_gt_pose else None,
            'original_first_timestamp': first_timestamp if first_timestamp else None,
            'normalization_applied': True if first_gt_pose else False,
            'pose_normalization': 'Transformed to start from origin (0,0,0)',
            'timestamp_normalization': 'Normalized to start from 0.000000000 seconds',
            'output_formats': {
                'euroc_csv': 'mav0/state_groundtruth_estimate0/data.csv',
                'tum_normalized': 'mav0/state_groundtruth_estimate0/groundtruth_normalized.txt',
                'tum_original': 'mav0/state_groundtruth_estimate0/groundtruth.txt'
            }
        }
        
        with open(os.path.join(gt_dir, "transformation_info.yaml"), 'w') as f:
            yaml.dump(transform_info, f, default_flow_style=False, indent=2)
        print(f"📝 Saved transformation information to transformation_info.yaml")
    
    # Create sensor configuration files
    print("⚙️ Creating sensor.yaml files...")
    create_sensor_yaml_files(mav0_dir, cam0_dir, cam1_dir, imu0_dir, gt_dir, camera_info_left, camera_info_right)
    
    print(f"\n🎉 Extraction complete! Data saved to: {mav0_dir}")
    print(f"\n📋 SUMMARY:")
    print(f"  📷 Camera images: {len(cam0_data)} left, {len(cam1_data)} right")
    print(f"  📐 IMU measurements: {len(imu_data)}")
    print(f"  🎯 Ground truth poses: {len(gt_data)}")
    print(f"  📊 Output formats:")
    print(f"    • EuRoC CSV: {gt_dir}/data.csv")
    print(f"    • TUM (normalized): {gt_dir}/groundtruth_normalized.txt")
    print(f"    • TUM (original): {gt_dir}/groundtruth.txt")
    print(f"\n✅ Ready for ORB-SLAM3 evaluation!")

def create_sensor_yaml_files(mav0_dir, cam0_dir, cam1_dir, imu0_dir, gt_dir, camera_info_left, camera_info_right):
    """Create individual sensor.yaml files for each sensor"""
    
    # Camera 0 (left) sensor.yaml
    cam0_config = {
        "sensor_type": "camera",
        "comment": "Left camera",
        "T_BS": {
            "data": [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
        },
        "rate_hz": 30.0,
        "resolution": [640, 480],
        "camera_model": "pinhole",
        "intrinsics": list(camera_info_left.K) if camera_info_left else [458.0, 458.0, 320.0, 240.0],
        "distortion_model": "radtan",
        "distortion_coefficients": list(camera_info_left.D) if camera_info_left and camera_info_left.D else [0.0, 0.0, 0.0, 0.0]
    }
    
    # Camera 1 (right) sensor.yaml
    cam1_config = {
        "sensor_type": "camera",
        "comment": "Right camera",
        "T_BS": {
            "data": [1.0, 0.0, 0.0, 0.162,  # Assuming baseline of 162mm
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
        },
        "rate_hz": 30.0,
        "resolution": [640, 480],
        "camera_model": "pinhole", 
        "intrinsics": list(camera_info_right.K) if camera_info_right else [458.0, 458.0, 320.0, 240.0],
        "distortion_model": "radtan",
        "distortion_coefficients": list(camera_info_right.D) if camera_info_right and camera_info_right.D else [0.0, 0.0, 0.0, 0.0]
    }
    
    # IMU sensor.yaml
    imu0_config = {
        "sensor_type": "imu",
        "comment": "IMU sensor",
        "T_BS": {
            "data": [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
        },
        "rate_hz": 200.0,
        "gyroscope_noise_density": 1.6968e-04,
        "gyroscope_random_walk": 1.9393e-05,
        "accelerometer_noise_density": 2.0000e-3,
        "accelerometer_random_walk": 3.0000e-3
    }
    
    # Ground truth sensor.yaml
    gt_config = {
        "sensor_type": "groundtruth",
        "comment": "Ground truth pose",
        "T_BS": {
            "data": [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
        },
        "rate_hz": 100.0
    }
    
    # Write individual sensor.yaml files
    cam0_sensor_path = os.path.join(os.path.dirname(cam0_dir), "sensor.yaml")
    with open(cam0_sensor_path, 'w') as f:
        yaml.dump(cam0_config, f, default_flow_style=False, indent=2)
    
    cam1_sensor_path = os.path.join(os.path.dirname(cam1_dir), "sensor.yaml")
    with open(cam1_sensor_path, 'w') as f:
        yaml.dump(cam1_config, f, default_flow_style=False, indent=2)
    
    imu0_sensor_path = os.path.join(imu0_dir, "sensor.yaml")
    with open(imu0_sensor_path, 'w') as f:
        yaml.dump(imu0_config, f, default_flow_style=False, indent=2)
    
    gt_sensor_path = os.path.join(gt_dir, "sensor.yaml")
    with open(gt_sensor_path, 'w') as f:
        yaml.dump(gt_config, f, default_flow_style=False, indent=2)
    
    print(f"Created sensor.yaml for cam0: {cam0_sensor_path}")
    print(f"Created sensor.yaml for cam1: {cam1_sensor_path}")
    print(f"Created sensor.yaml for imu0: {imu0_sensor_path}")
    print(f"Created sensor.yaml for ground truth: {gt_sensor_path}")

def validate_extraction_outputs(output_dir):
    """Validate that all expected outputs are correctly generated"""
    print("\n🔍 Validating extraction outputs...")
    
    mav0_dir = os.path.join(output_dir, "mav0")
    issues = []
    
    # Check directory structure
    required_dirs = [
        os.path.join(mav0_dir, "cam0"),
        os.path.join(mav0_dir, "cam1"), 
        os.path.join(mav0_dir, "imu0"),
        os.path.join(mav0_dir, "state_groundtruth_estimate0")
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_path}")
    
    # Check required files
    required_files = [
        os.path.join(mav0_dir, "cam0", "data.csv"),
        os.path.join(mav0_dir, "cam1", "data.csv"),
        os.path.join(mav0_dir, "imu0", "data.csv"),
        os.path.join(mav0_dir, "state_groundtruth_estimate0", "data.csv"),
        os.path.join(mav0_dir, "state_groundtruth_estimate0", "groundtruth_normalized.txt"),
        os.path.join(mav0_dir, "state_groundtruth_estimate0", "groundtruth.txt")
    ]
    
    file_sizes = {}
    for file_path in required_files:
        if not os.path.exists(file_path):
            issues.append(f"Missing file: {file_path}")
        else:
            file_sizes[os.path.basename(file_path)] = os.path.getsize(file_path)
    
    # Validate ground truth file consistency
    gt_normalized_path = os.path.join(mav0_dir, "state_groundtruth_estimate0", "groundtruth_normalized.txt")
    gt_original_path = os.path.join(mav0_dir, "state_groundtruth_estimate0", "groundtruth.txt")
    
    if os.path.exists(gt_normalized_path) and os.path.exists(gt_original_path):
        # Count lines in both files
        with open(gt_normalized_path, 'r') as f:
            normalized_lines = len(f.readlines())
        with open(gt_original_path, 'r') as f:
            original_lines = len(f.readlines())
            
        if normalized_lines != original_lines:
            issues.append(f"Ground truth file line count mismatch: normalized={normalized_lines}, original={original_lines}")
        
        # Check first timestamp in normalized file is ~0
        with open(gt_normalized_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                first_timestamp = float(first_line.split()[0])
                if abs(first_timestamp) > 0.001:  # Should be very close to 0
                    issues.append(f"Normalized timestamp doesn't start near 0: {first_timestamp}")
    
    # Print validation results
    if issues:
        print("❌ Validation issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ All outputs validated successfully!")
        print("\n📊 File sizes:")
        for filename, size in file_sizes.items():
            size_kb = size / 1024
            if size_kb > 1024:
                print(f"  • {filename}: {size_kb/1024:.1f} MB")
            else:
                print(f"  • {filename}: {size_kb:.1f} KB")
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Extract ROS bag data to mav0 format with integrated ground truth processing',
        epilog="""
Examples:
  python3 extract_bag_to_mav0.py lunar_dataset.bag ./output
  python3 extract_bag_to_mav0.py /path/to/data.bag /path/to/output

Output:
  Creates mav0/ directory structure with EuRoC format data
  Generates groundtruth_normalized.txt ready for ORB-SLAM3 evaluation
  Generates groundtruth.txt with original timestamps for reference
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('bag_path', help='Path to the ROS bag file')
    parser.add_argument('output_dir', help='Output directory for mav0 data')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation checks after extraction')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bag_path):
        print(f"❌ Error: Bag file {args.bag_path} not found!")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting streamlined ROS bag extraction...")
    print("=" * 60)
    
    # Extract data
    extract_bag_data(args.bag_path, args.output_dir)
    
    # Validate outputs if requested
    if args.validate:
        validation_passed = validate_extraction_outputs(args.output_dir)
        if not validation_passed:
            return 1
    
    print("\n" + "=" * 60)
    print("🎉 Extraction completed successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print("\n🎯 Next steps:")
    print("  1. Use mav0/state_groundtruth_estimate0/groundtruth_normalized.txt for ORB-SLAM3 evaluation")
    print("  2. Run: python3 visualize_trajectories_directly.py")
    print("  3. Dataset is ready for SLAM processing!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 
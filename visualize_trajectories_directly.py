#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import sys
import glob
import argparse
from pathlib import Path

def read_tum_trajectory(filepath, max_poses=None):
    """Read trajectory from TUM format file"""
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return None
        
    data = []
    count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 8:
                        timestamp = float(parts[0])
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        data.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
                        count += 1
                        if max_poses and count >= max_poses:
                            break
        
        if not data:
            print(f"⚠️  No valid trajectory data found in: {filepath}")
            return None
            
        df = pd.DataFrame(data, columns=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        print(f"✅ Loaded {len(df)} poses from: {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def discover_trajectory_files(mav0_path):
    """Discover all trajectory files in the mav0 output structure"""
    mav0_path = Path(mav0_path)
    
    # Ground truth file
    gt_file = mav0_path / "state_groundtruth_estimate0" / "groundtruth_normalized.txt"
    
    # Discover output directories and their trajectory files
    output_dir = mav0_path / "output"
    trajectories = {}
    
    if not output_dir.exists():
        print(f"⚠️  Output directory not found: {output_dir}")
        return str(gt_file), trajectories
    
    print(f"🔍 Discovering trajectory files in: {output_dir}")
    
    # Look for subdirectories in output/
    for method_dir in output_dir.iterdir():
        if method_dir.is_dir():
            method_name = method_dir.name
            print(f"📁 Found method directory: {method_name}")
            
            # Look for trajectory files in this method directory
            trajectory_files = []
            
            # Common trajectory file patterns
            patterns = [
                "CameraTrajectory*.txt",
                "KeyFrameTrajectory*.txt", 
                "*Trajectory*.txt",
                "trajectory*.txt",
                "*.txt"
            ]
            
            for pattern in patterns:
                matches = list(method_dir.glob(pattern))
                for match in matches:
                    if match.is_file() and match.name.endswith('.txt'):
                        trajectory_files.append(match)
            
            # Remove duplicates and sort
            trajectory_files = sorted(list(set(trajectory_files)))
            
            if trajectory_files:
                print(f"  📄 Found trajectory files:")
                for traj_file in trajectory_files:
                    print(f"    - {traj_file.name}")
                
                # Store all trajectory files for this method
                trajectories[method_name] = trajectory_files
            else:
                print(f"  ⚠️  No trajectory files found in {method_name}")
    
    return str(gt_file), trajectories

def find_best_coordinate_transformation(gt_trajectory, comparison_trajectories):
    """Find the best coordinate transformation including sign flips"""
    
    print("\n🔍 Testing different coordinate transformations and sign flips...")
    
    if gt_trajectory is None or len(comparison_trajectories) == 0:
        print("❌ No trajectories to analyze")
        return {}, 'Original'
    
    gt_pos = gt_trajectory[['tx', 'ty', 'tz']].values
    
    # Use the first trajectory for transformation analysis
    first_traj = list(comparison_trajectories.values())[0]
    if isinstance(first_traj, list):
        first_traj = first_traj[0]  # Use first file if multiple
    comparison_pos = first_traj[['tx', 'ty', 'tz']].values
    
    # Try different transformations with sign flips
    transformations = {
        'Original': gt_pos,
        'X→Y, Y→Z, Z→X': gt_pos[:, [1, 2, 0]], 
        'X→Y, Y→Z, Z→X, flip_X': np.column_stack([-gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 0]]),
        'X→Y, Y→Z, Z→X, flip_Y': np.column_stack([gt_pos[:, 1], -gt_pos[:, 2], gt_pos[:, 0]]),
        'X→Y, Y→Z, Z→X, flip_Z': np.column_stack([gt_pos[:, 1], gt_pos[:, 2], -gt_pos[:, 0]]),
        'X→Y, Y→Z, Z→X, flip_XY': np.column_stack([-gt_pos[:, 1], -gt_pos[:, 2], gt_pos[:, 0]]),
        'X→Y, Y→Z, Z→X, flip_XZ': np.column_stack([-gt_pos[:, 1], gt_pos[:, 2], -gt_pos[:, 0]]),
        'X→Y, Y→Z, Z→X, flip_YZ': np.column_stack([gt_pos[:, 1], -gt_pos[:, 2], -gt_pos[:, 0]]),
        'X→Y, Y→Z, Z→X, flip_XYZ': np.column_stack([-gt_pos[:, 1], -gt_pos[:, 2], -gt_pos[:, 0]]),
    }
    
    # Calculate similarity metrics for each transformation
    results = {}
    
    for name, gt_transformed in transformations.items():
        # Calculate ranges
        gt_ranges = np.ptp(gt_transformed, axis=0)
        comp_ranges = np.ptp(comparison_pos, axis=0)
        
        # Calculate similarity score (how well ranges match)
        range_ratios = np.minimum(gt_ranges, comp_ranges) / np.maximum(gt_ranges, comp_ranges)
        similarity_score = np.mean(range_ratios)
        
        # Calculate shape correlation
        if len(gt_transformed) > len(comparison_pos):
            indices = np.linspace(0, len(gt_transformed)-1, len(comparison_pos), dtype=int)
            gt_sampled = gt_transformed[indices]
            comp_sampled = comparison_pos
        else:
            gt_sampled = gt_transformed
            indices = np.linspace(0, len(comparison_pos)-1, len(gt_transformed), dtype=int)
            comp_sampled = comparison_pos[indices]
        
        try:
            corr_x = np.corrcoef(gt_sampled[:, 0], comp_sampled[:, 0])[0, 1]
            corr_y = np.corrcoef(gt_sampled[:, 1], comp_sampled[:, 1])[0, 1]
            
            # Strongly prefer positive correlations (correct orientation) over negative ones (flipped)
            # Give full weight to positive correlations, very little to negative ones
            pos_corr_x = max(0, corr_x) + 0.05 * abs(min(0, corr_x))
            pos_corr_y = max(0, corr_y) + 0.05 * abs(min(0, corr_y))
            
            # Additional penalty if both X and Y are negatively correlated (complete flip)
            if corr_x < 0 and corr_y < 0:
                flip_penalty = 0.3  # Reduce score significantly for complete flips
                pos_corr_x *= (1 - flip_penalty)
                pos_corr_y *= (1 - flip_penalty)
            
            shape_correlation = (pos_corr_x + pos_corr_y) / 2
        except:
            corr_x, corr_y = 0, 0
            shape_correlation = 0
        
        # Combined score
        combined_score = (similarity_score + shape_correlation) / 2
        
        results[name] = {
            'gt_ranges': gt_ranges,
            'comp_ranges': comp_ranges,
            'similarity_score': similarity_score,
            'shape_correlation': shape_correlation,
            'combined_score': combined_score,
            'transformed_pos': gt_transformed,
            'corr_x': corr_x,
            'corr_y': corr_y
        }
        
        print(f"\n🔍 {name}:")
        print(f"  GT Ranges: X={gt_ranges[0]:.3f}, Y={gt_ranges[1]:.3f}, Z={gt_ranges[2]:.3f}")
        print(f"  Comp Ranges: X={comp_ranges[0]:.3f}, Y={comp_ranges[1]:.3f}, Z={comp_ranges[2]:.3f}")
        print(f"  Range Similarity: {similarity_score:.3f}")
        print(f"  Raw Correlations: X:{corr_x:.3f}, Y:{corr_y:.3f}")
        print(f"  Oriented Correlation: {shape_correlation:.3f}")
        print(f"  Combined Score: {combined_score:.3f}")
    
    # Find best transformation
    best_transform = max(results.keys(), key=lambda k: results[k]['combined_score'])
    
    print(f"\n🏆 Best transformation: {best_transform}")
    print(f"   Combined score: {results[best_transform]['combined_score']:.3f}")
    
    return results, best_transform

def apply_coordinate_transformation(df, transform_name):
    """Apply coordinate transformation to dataframe"""
    if df is None:
        return None
        
    df_transformed = df.copy()
    
    orig_x = df['tx'].values
    orig_y = df['ty'].values  
    orig_z = df['tz'].values
    
    if transform_name == 'Original':
        pass
    elif transform_name == 'X→Y, Y→Z, Z→X':
        df_transformed['tx'] = orig_y  
        df_transformed['ty'] = orig_z  
        df_transformed['tz'] = orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_X':
        df_transformed['tx'] = -orig_y  
        df_transformed['ty'] = orig_z  
        df_transformed['tz'] = orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_Y':
        df_transformed['tx'] = orig_y  
        df_transformed['ty'] = -orig_z  
        df_transformed['tz'] = orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_Z':
        df_transformed['tx'] = orig_y  
        df_transformed['ty'] = orig_z  
        df_transformed['tz'] = -orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_XY':
        df_transformed['tx'] = -orig_y  
        df_transformed['ty'] = -orig_z  
        df_transformed['tz'] = orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_XZ':
        df_transformed['tx'] = -orig_y  
        df_transformed['ty'] = orig_z  
        df_transformed['tz'] = -orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_YZ':
        df_transformed['tx'] = orig_y  
        df_transformed['ty'] = -orig_z  
        df_transformed['tz'] = -orig_x
    elif transform_name == 'X→Y, Y→Z, Z→X, flip_XYZ':
        df_transformed['tx'] = -orig_y  
        df_transformed['ty'] = -orig_z  
        df_transformed['tz'] = -orig_x
    
    return df_transformed

def create_comprehensive_comparison(mav0_path):
    """Create comprehensive trajectory comparison from mav0 structure"""
    
    print(f"🎯 COMPREHENSIVE MAV0 TRAJECTORY ANALYSIS")
    print("="*60)
    print(f"📁 MAV0 Path: {mav0_path}")
    
    # Discover all trajectory files
    gt_file, trajectory_files = discover_trajectory_files(mav0_path)
    
    # Load ground truth
    print(f"\n📍 Loading ground truth from: {gt_file}")
    gt_original = read_tum_trajectory(gt_file)
    
    if gt_original is None:
        print("❌ Could not load ground truth. Exiting.")
        return
    
    # Load all trajectory files
    all_trajectories = {}
    trajectory_data = {}
    
    for method_name, files in trajectory_files.items():
        print(f"\n📊 Loading trajectories for method: {method_name}")
        method_trajectories = {}
        
        for traj_file in files:
            traj_name = traj_file.stem  # filename without extension
            traj_data = read_tum_trajectory(str(traj_file))
            
            if traj_data is not None:
                method_trajectories[traj_name] = traj_data
                
                # Use the first trajectory file as primary for this method
                if len(method_trajectories) == 1:
                    trajectory_data[method_name] = traj_data
        
        all_trajectories[method_name] = method_trajectories
    
    if not trajectory_data:
        print("❌ No valid trajectory files found. Exiting.")
        return
    
    # Find best coordinate transformation
    print(f"\n🔄 Finding optimal coordinate transformation...")
    results, best_transform = find_best_coordinate_transformation(gt_original, trajectory_data)
    
    if not results:
        print("❌ Could not determine coordinate transformation. Using original.")
        best_transform = 'Original'
    
    # Apply transformation to ground truth
    print(f"\n🔄 Applying coordinate transformation: {best_transform}")
    gt = apply_coordinate_transformation(gt_original, best_transform)
    
    # Create visualization
    print(f"\n🎨 Creating comprehensive visualization...")
    
    # Determine subplot layout based on number of methods
    n_methods = len(trajectory_data)
    fig_width = max(20, 5 * n_methods)
    fig_height = 15
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Color mapping for different methods
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(trajectory_data.keys())}
    
    # 1. Main XY trajectory comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(gt['tx'].values, gt['ty'].values, 'k-', linewidth=3, label=f'Ground Truth ({len(gt)} poses)', alpha=0.8)
    
    for method_name, traj in trajectory_data.items():
        color = method_colors[method_name]
        ax1.plot(traj['tx'].values, traj['ty'].values, color=color, linewidth=2, 
                label=f'{method_name} ({len(traj)} poses)', alpha=0.8)
        
        # Mark start points
        ax1.plot(traj['tx'].iloc[0], traj['ty'].iloc[0], 'o', color=color, 
                markersize=6, alpha=0.8)
    
    # Mark GT start
    ax1.plot(gt['tx'].iloc[0], gt['ty'].iloc[0], 'ko', markersize=8, label='GT Start')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'XY Trajectory Comparison\n({best_transform})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. XZ trajectory comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(gt['tx'].values, gt['tz'].values, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    
    for method_name, traj in trajectory_data.items():
        color = method_colors[method_name]
        ax2.plot(traj['tx'].values, traj['tz'].values, color=color, linewidth=2, 
                label=method_name, alpha=0.8)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('XZ Trajectory Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Position vs Time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(gt['timestamp'].values, gt['tx'].values, 'k-', linewidth=2, label='GT X', alpha=0.7)
    ax3.plot(gt['timestamp'].values, gt['ty'].values, 'k--', linewidth=2, label='GT Y', alpha=0.7)
    
    for method_name, traj in trajectory_data.items():
        color = method_colors[method_name]
        ax3.plot(traj['timestamp'].values, traj['tx'].values, color=color, linewidth=1.5, 
                label=f'{method_name} X', alpha=0.8)
        ax3.plot(traj['timestamp'].values, traj['ty'].values, '--', color=color, linewidth=1.5, 
                label=f'{method_name} Y', alpha=0.8)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D trajectory
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    ax4.plot(gt['tx'].values, gt['ty'].values, gt['tz'].values, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    
    for method_name, traj in trajectory_data.items():
        color = method_colors[method_name]
        ax4.plot(traj['tx'].values, traj['ty'].values, traj['tz'].values, color=color, linewidth=2, 
                label=method_name, alpha=0.8)
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Trajectory Comparison')
    ax4.legend()
    
    # 5. Coordinate range comparison
    ax5 = plt.subplot(2, 3, 5)
    
    gt_ranges = np.array([gt['tx'].values.max() - gt['tx'].values.min(), 
                         gt['ty'].values.max() - gt['ty'].values.min(), 
                         gt['tz'].values.max() - gt['tz'].values.min()])
    
    x_pos = np.arange(3)
    width = 0.2
    
    # Plot ground truth
    ax5.bar(x_pos - width * (len(trajectory_data)/2), gt_ranges, width, 
           label='Ground Truth', alpha=0.8, color='black')
    
    # Plot each method
    for i, (method_name, traj) in enumerate(trajectory_data.items()):
        traj_ranges = np.array([traj['tx'].values.max() - traj['tx'].values.min(), 
                               traj['ty'].values.max() - traj['ty'].values.min(), 
                               traj['tz'].values.max() - traj['tz'].values.min()])
        
        offset = width * (i - len(trajectory_data)/2 + 0.5)
        color = method_colors[method_name]
        ax5.bar(x_pos + offset, traj_ranges, width, 
               label=method_name, alpha=0.8, color=color)
    
    ax5.set_xlabel('Coordinate Axis')
    ax5.set_ylabel('Range (m)')
    ax5.set_title('Coordinate Range Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['X', 'Y', 'Z'])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate and display statistics
    stats_text = f"📊 TRAJECTORY STATISTICS\n\n"
    stats_text += f"Ground Truth ({best_transform}):\n"
    
    gt_path = np.sum(np.sqrt(np.diff(gt['tx'].values)**2 + np.diff(gt['ty'].values)**2 + np.diff(gt['tz'].values)**2))
    gt_duration = gt['timestamp'].values.max() - gt['timestamp'].values.min()
    stats_text += f"  Path: {gt_path:.1f}m, Duration: {gt_duration:.1f}s\n"
    stats_text += f"  Poses: {len(gt)}\n\n"
    
    for method_name, traj in trajectory_data.items():
        path_length = np.sum(np.sqrt(np.diff(traj['tx'].values)**2 + np.diff(traj['ty'].values)**2 + np.diff(traj['tz'].values)**2))
        duration = traj['timestamp'].values.max() - traj['timestamp'].values.min()
        
        stats_text += f"{method_name}:\n"
        stats_text += f"  Path: {path_length:.1f}m ({path_length/gt_path*100:.1f}% of GT)\n"
        stats_text += f"  Duration: {duration:.1f}s ({duration/gt_duration*100:.1f}% of GT)\n"
        stats_text += f"  Poses: {len(traj)}\n\n"
    
    # Add alignment quality if available
    if best_transform in results:
        result = results[best_transform]
        stats_text += f"🎯 ALIGNMENT QUALITY:\n"
        stats_text += f"  Range Similarity: {result['similarity_score']:.1%}\n"
        stats_text += f"  Shape Correlation: {result['shape_correlation']:.1%}\n"
        stats_text += f"  Combined Score: {result['combined_score']:.1%}\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"mav0_trajectory_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive comparison: {output_file}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("🎯 MAV0 TRAJECTORY COMPARISON SUMMARY")
    print("="*60)
    print(f"📁 MAV0 Path: {mav0_path}")
    print(f"🔧 Coordinate Transform: {best_transform}")
    
    if best_transform in results:
        result = results[best_transform]
        print(f"📊 Alignment Quality: {result['combined_score']:.1%}")
    
    print(f"\n📊 Methods Analyzed:")
    for method_name, files in all_trajectories.items():
        print(f"  🔹 {method_name}: {len(files)} trajectory files")
        for filename, traj in files.items():
            if traj is not None:
                print(f"    - {filename}: {len(traj)} poses")
    
    print(f"\n✅ Analysis complete! Check {output_file} for visualization.")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive MAV0 trajectory analysis and comparison')
    parser.add_argument('mav0_path', help='Path to the mav0 folder containing state_groundtruth_estimate0 and output subdirectories')
    parser.add_argument('--max-poses', type=int, help='Maximum number of poses to load per trajectory (for testing)')
    
    args = parser.parse_args()
    
    # Check if mav0 path exists
    if not os.path.exists(args.mav0_path):
        print(f"❌ MAV0 path does not exist: {args.mav0_path}")
        sys.exit(1)
    
    # Run comprehensive analysis
    create_comprehensive_comparison(args.mav0_path)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, show help and try to auto-detect
        print("🎯 MAV0 TRAJECTORY ANALYZER")
        print("="*40)
        print("Usage: python3 visualize_trajectories_directly.py <mav0_path>")
        print("Example: python3 visualize_trajectories_directly.py ./my_dataset/mav0")
        print()
        
        # Try to auto-detect mav0 folders in current directory
        possible_paths = []
        for item in os.listdir('.'):
            if os.path.isdir(item):
                # Check if it looks like a mav0 folder
                if (os.path.exists(os.path.join(item, 'state_groundtruth_estimate0')) or 
                    item.endswith('mav0') or 
                    os.path.exists(os.path.join(item, 'output'))):
                    possible_paths.append(item)
        
        if possible_paths:
            print("🔍 Auto-detected possible mav0 folders:")
            for path in possible_paths:
                print(f"  - {path}")
            print()
            print("Run with one of these paths as argument.")
        
        sys.exit(1)
    
    main() 
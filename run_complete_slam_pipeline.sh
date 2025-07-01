#!/bin/bash

# ================================================
# Complete SLAM Pipeline Automation Script
# ================================================
# 
# This script automates the entire pipeline:
# 1. Extract ROS bag to mav0 format
# 2. Synchronize timestamps and stereo pairs for ORB-SLAM3
# 3. Run regular ORB-SLAM3
# 4. Run FoundationStereo-enhanced ORB-SLAM3
# 5. Generate comprehensive trajectory comparison
#
# Usage: ./run_complete_slam_pipeline.sh <bag_file> [options]
# ================================================

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="outputs/slam_results_$(date +%Y%m%d_%H%M%S)"
VOCABULARY_PATH="Vocabulary/ORBvoc.txt"
CONFIG_PATH="Examples/Stereo/Lunar.yaml"
TIMESTAMP_FILE="Examples/Stereo/EuRoC_TimeStamps/LUNAR_synchronized.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        log_error "Required file not found: $1"
        exit 1
    fi
}

# Function to check if directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        log_error "Required directory not found: $1"
        exit 1
    fi
}

# Function to show help
show_help() {
    cat << EOF
${BOLD}Complete SLAM Pipeline Automation Script${NC}

${BOLD}USAGE:${NC}
    $0 <bag_file> [OPTIONS]

${BOLD}ARGUMENTS:${NC}
    bag_file            Path to the ROS bag file to process

${BOLD}OPTIONS:${NC}
    -o, --output DIR    Output directory (default: slam_results_YYYYMMDD_HHMMSS)
    -s, --skip-extract  Skip ROS bag extraction (use existing mav0 data)
    -r, --regular-only  Run only regular ORB-SLAM3 (skip FoundationStereo)
    -f, --fs-only       Run only FoundationStereo SLAM (skip regular)
    -v, --verbose       Enable verbose output
    -h, --help          Show this help message

${BOLD}EXAMPLES:${NC}
    # Process complete pipeline
    $0 /path/to/dataset.bag

    # Use custom output directory
    $0 /path/to/dataset.bag -o my_results

    # Skip extraction (use existing mav0 data)
    $0 /path/to/dataset.bag -s -o existing_data

    # Run only regular ORB-SLAM3
    $0 /path/to/dataset.bag -r

    # Run only FoundationStereo
    $0 /path/to/dataset.bag -f

${BOLD}REQUIREMENTS:${NC}
    - ORB-SLAM3 built and ready
    - Python 3 with required packages
    - extract_bag_to_mav0.py
    - visualize_trajectories_directly.py

${BOLD}OUTPUT STRUCTURE:${NC}
    output_dir/
    ├── mav0/                          # Extracted dataset
    │   ├── state_groundtruth_estimate0/
    │   └── output/
    │       ├── orbslam3/              # Regular SLAM results
    │       └── foundation_stereo/      # FoundationStereo results
    ├── logs/                          # Execution logs
    └── mav0_trajectory_comparison.png # Final visualization

EOF
}

# Function to extract ROS bag data
extract_bag_data() {
    local bag_file="$1"
    local output_dir="$2"
    
    log_step "STEP 1: Extracting ROS Bag Data"
    
    log_info "Input bag: $bag_file"
    log_info "Output directory: $output_dir"
    
    # Check if extract script exists
    check_file "extract_bag_to_mav0.py"
    
    # Run extraction
    log_info "Running bag extraction..."
    if [ "$VERBOSE" = true ]; then
        python3 extract_bag_to_mav0.py "$bag_file" "$output_dir" --validate
    else
        python3 extract_bag_to_mav0.py "$bag_file" "$output_dir" --validate > "$output_dir/logs/extraction.log" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        log_success "ROS bag extraction completed successfully"
        
        # Check if mav0 directory was created
        local mav0_dir="$output_dir/mav0"
        check_directory "$mav0_dir"
        
        # Show extraction summary
        local gt_file="$mav0_dir/state_groundtruth_estimate0/groundtruth_normalized.txt"
        if [ -f "$gt_file" ]; then
            local pose_count=$(wc -l < "$gt_file")
            log_info "Extracted $pose_count ground truth poses"
        fi
        
        local cam0_dir="$mav0_dir/cam0/data"
        if [ -d "$cam0_dir" ]; then
            local image_count=$(ls "$cam0_dir"/*.png 2>/dev/null | wc -l)
            log_info "Extracted $image_count stereo image pairs"
        fi
        
    else
        log_error "ROS bag extraction failed. Check $output_dir/logs/extraction.log for details."
        exit 1
    fi
}

# Function to synchronize timestamps and stereo pairs for ORB-SLAM3 compatibility
synchronize_stereo_timestamps() {
    local mav0_dir="$1"
    local output_dir="$2"
    
    log_step "TIMESTAMP & STEREO SYNCHRONIZATION"
    
    log_info "Synchronizing timestamps and stereo pairs for ORB-SLAM3..."
    log_info "  Converting image names to decimal timestamp format"
    log_info "  Synchronizing stereo pairs with identical timestamps"
    
    # Create timestamp directory if it doesn't exist
    mkdir -p "$(dirname "$TIMESTAMP_FILE")"
    
    # Step 1: Generate timestamp file from cam0 image names
    log_info "Step 1: Generating timestamp file from image names..."
    ls "$mav0_dir/cam0/data"/*.png | sed 's/.*\///; s/\.png$//' | awk '{printf "%.9f\n", $1/1e9}' > "$TIMESTAMP_FILE"
    local timestamp_count=$(wc -l < "$TIMESTAMP_FILE")
    log_info "Generated $timestamp_count synchronized timestamps: $TIMESTAMP_FILE"
    
    # Step 2: Rename cam0 images to decimal timestamp format
    log_info "Step 2: Converting cam0 images to decimal timestamp format..."
    cd "$mav0_dir/cam0/data"
    for file in *.png; do
        if [ -f "$file" ]; then
            ts=${file%.png}
            new_name=$(echo $ts | awk '{printf "%.9f.png", $1/1e9}')
            if [ "$file" != "$new_name" ]; then
                mv "$file" "$new_name"
            fi
        fi
    done
    cd - > /dev/null
    
    # Step 3: Rename cam1 images to decimal timestamp format  
    log_info "Step 3: Converting cam1 images to decimal timestamp format..."
    cd "$mav0_dir/cam1/data"
    for file in *.png; do
        if [ -f "$file" ]; then
            ts=${file%.png}
            new_name=$(echo $ts | awk '{printf "%.9f.png", $1/1e9}')
            if [ "$file" != "$new_name" ]; then
                mv "$file" "$new_name"
            fi
        fi
    done
    cd - > /dev/null
    
    # Step 4: Synchronize cam1 images with cam0 timestamps
    log_info "Step 4: Synchronizing stereo pairs with identical timestamps..."
    cd "$mav0_dir"
    paste <(ls cam0/data/*.png | sort) <(ls cam1/data/*.png | sort) | while read cam0_file cam1_file; do
        if [ -f "$cam0_file" ] && [ -f "$cam1_file" ]; then
            cam0_name=$(basename "$cam0_file")
            cam1_current=$(basename "$cam1_file")
            if [ "$cam0_name" != "$cam1_current" ]; then
                mv "$cam1_file" "cam1/data/$cam0_name"
            fi
        fi
    done
    cd - > /dev/null
    
    # Verify synchronization
    local cam0_count=$(ls "$mav0_dir/cam0/data"/*.png 2>/dev/null | wc -l)
    local cam1_count=$(ls "$mav0_dir/cam1/data"/*.png 2>/dev/null | wc -l) 
    
    if [ "$cam0_count" -eq "$cam1_count" ]; then
        log_success "Stereo synchronization completed: $cam0_count synchronized pairs"
    else
        log_warning "Stereo count mismatch: cam0=$cam0_count, cam1=$cam1_count"
    fi
}

# Function to generate synchronized timestamp file
generate_timestamps() {
    local mav0_dir="$1"
    local output_dir="$2"
    
    log_info "Verifying synchronized timestamp file..."
    
    # Check if timestamp file exists and has correct count
    if [ -f "$TIMESTAMP_FILE" ]; then
        local timestamp_count=$(wc -l < "$TIMESTAMP_FILE")
        local image_count=$(ls "$mav0_dir/cam0/data"/*.png 2>/dev/null | wc -l)
        
        if [ "$timestamp_count" -eq "$image_count" ]; then
            log_info "Timestamp file verified: $timestamp_count timestamps match $image_count images"
            return 0
        fi
    fi
    
    # Generate timestamp file if needed
    log_info "Generating timestamp file from synchronized images..."
    mkdir -p "$(dirname "$TIMESTAMP_FILE")"
    ls "$mav0_dir/cam0/data"/*.png | sed 's/.*\///; s/\.png$//' > "$TIMESTAMP_FILE"
    local timestamp_count=$(wc -l < "$TIMESTAMP_FILE")
    log_info "Generated $timestamp_count timestamps: $TIMESTAMP_FILE"
}

# Function to run regular ORB-SLAM3
run_regular_slam() {
    local mav0_dir="$1"
    local output_dir="$2"
    
    log_step "STEP 3: Running Regular ORB-SLAM3"
    
    # Check required files
    check_file "$VOCABULARY_PATH"
    check_file "$CONFIG_PATH"
    check_file "$TIMESTAMP_FILE"
    check_file "Examples/Stereo/stereo_euroc"
    
    log_info "Running regular ORB-SLAM3..."
    log_info "  Vocabulary: $VOCABULARY_PATH"
    log_info "  Config: $CONFIG_PATH"
    log_info "  Dataset: $mav0_dir"
    log_info "  Timestamps: $TIMESTAMP_FILE"
    
    # Run SLAM
    local log_file="$output_dir/logs/regular_slam.log"
    if [ "$VERBOSE" = true ]; then
        ./Examples/Stereo/stereo_euroc \
            "$VOCABULARY_PATH" \
            "$CONFIG_PATH" \
            "$output_dir" \
            "$TIMESTAMP_FILE" \
            2>&1 | tee "$log_file"
    else
        ./Examples/Stereo/stereo_euroc \
            "$VOCABULARY_PATH" \
            "$CONFIG_PATH" \
            "$output_dir" \
            "$TIMESTAMP_FILE" \
            > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Regular ORB-SLAM3 completed successfully"
    else
        log_error "Regular ORB-SLAM3 failed with exit code $exit_code"
        log_error "Check $log_file for details"
        return 1
    fi
    
    # Check if output files were created
    local output_orbslam_dir="$mav0_dir/output/orbslam3"
    if [ -d "$output_orbslam_dir" ]; then
        local trajectory_file="$output_orbslam_dir/CameraTrajectory.txt"
        if [ -f "$trajectory_file" ]; then
            local pose_count=$(wc -l < "$trajectory_file")
            log_success "Regular SLAM trajectory saved: $pose_count poses"
        else
            log_warning "Regular SLAM trajectory file not found"
        fi
    else
        log_warning "Regular SLAM output directory not created"
    fi
}

# Function to run FoundationStereo SLAM
run_foundationstereo_slam() {
    local mav0_dir="$1"
    local output_dir="$2"
    
    log_step "STEP 4: Running FoundationStereo-Enhanced ORB-SLAM3"
    
    # Check required files
    check_file "Examples/Stereo/stereo_euroc_foundationstereo"
    
    log_info "Running FoundationStereo-enhanced ORB-SLAM3..."
    log_info "  Using deep learning stereo matching"
    log_info "  Enhanced depth estimation"
    
    # Run FoundationStereo SLAM
    local log_file="$output_dir/logs/foundationstereo_slam.log"
    if [ "$VERBOSE" = true ]; then
        ./Examples/Stereo/stereo_euroc_foundationstereo \
            "$VOCABULARY_PATH" \
            "$CONFIG_PATH" \
            "$output_dir" \
            "$TIMESTAMP_FILE" \
            2>&1 | tee "$log_file"
    else
        ./Examples/Stereo/stereo_euroc_foundationstereo \
            "$VOCABULARY_PATH" \
            "$CONFIG_PATH" \
            "$output_dir" \
            "$TIMESTAMP_FILE" \
            > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "FoundationStereo SLAM completed successfully"
    else
        log_error "FoundationStereo SLAM failed with exit code $exit_code"
        log_error "Check $log_file for details"
        return 1
    fi
    
    # Check if output files were created - try both directory name variants
    local output_fs_dir1="$mav0_dir/output/foundation_stereo"
    local output_fs_dir2="$mav0_dir/output/foundationstereo"
    local output_fs_dir=""
    
    if [ -d "$output_fs_dir1" ]; then
        output_fs_dir="$output_fs_dir1"
    elif [ -d "$output_fs_dir2" ]; then
        output_fs_dir="$output_fs_dir2"
    fi
    
    if [ -n "$output_fs_dir" ]; then
        log_info "FoundationStereo output directory found: $output_fs_dir"
        
        # Check for trajectory files with different possible names
        local trajectory_file=""
        for possible_name in "CameraTrajectory_FoundationStereo.txt" "CameraTrajectory.txt" "trajectory.txt"; do
            if [ -f "$output_fs_dir/$possible_name" ]; then
                trajectory_file="$output_fs_dir/$possible_name"
                break
            fi
        done
        
        if [ -n "$trajectory_file" ]; then
            local pose_count=$(wc -l < "$trajectory_file")
            log_success "FoundationStereo trajectory saved: $pose_count poses in $trajectory_file"
        else
            log_warning "FoundationStereo trajectory file not found. Available files:"
            ls -la "$output_fs_dir"/ | head -10
        fi
    else
        log_warning "FoundationStereo output directory not created. Checking available output directories:"
        ls -la "$mav0_dir/output/" 2>/dev/null || log_warning "No output directory found"
    fi
}

# Function to generate visualization
generate_visualization() {
    local mav0_dir="$1"
    local output_dir="$2"
    
    log_step "STEP 5: Generating Comprehensive Trajectory Visualization"
    
    # Check if visualization script exists
    check_file "visualize_trajectories_directly.py"
    
    log_info "Creating trajectory comparison visualization..."
    log_info "  Analyzing coordinate transformations"
    log_info "  Comparing all SLAM methods"
    log_info "  Generating comprehensive plots"
    
    # Ensure logs directory exists
    mkdir -p "$output_dir/logs"
    
    # Verify mav0 directory exists
    if [ ! -d "$mav0_dir" ]; then
        log_error "MAV0 directory not found: $mav0_dir"
        return 1
    fi
    
    log_info "MAV0 directory verified: $mav0_dir"
    
    # Change to output directory to save visualization there
    cd "$output_dir"
    
    # Run visualization with relative path (since we're now in output_dir)
    local log_file="logs/visualization.log"  # Use relative path since we're in output_dir
    log_info "Running visualization with log file: $output_dir/$log_file"
    
    if [ "$VERBOSE" = true ]; then
        python3 "$SCRIPT_DIR/visualize_trajectories_directly.py" "mav0" 2>&1 | tee "$log_file"
    else
        python3 "$SCRIPT_DIR/visualize_trajectories_directly.py" "mav0" > "$log_file" 2>&1
    fi
    
    local viz_exit_code=$?
    
    # Return to original directory
    cd "$SCRIPT_DIR"
    
    if [ $viz_exit_code -eq 0 ]; then
        log_success "Trajectory visualization completed successfully"
        
        # Check if visualization was created
        local viz_file="$output_dir/mav0_trajectory_comparison.png"
        if [ -f "$viz_file" ]; then
            log_success "Visualization saved: $viz_file"
        else
            log_warning "Visualization file not found"
        fi
    else
        log_error "Trajectory visualization failed"
        log_error "Check $output_dir/$log_file for details"
        return 1
    fi
}

# Function to show final summary
show_summary() {
    local output_dir="$1"
    local mav0_dir="$output_dir/mav0"
    
    log_step "PIPELINE SUMMARY"
    
    log_info "Output directory: $output_dir"
    
    # Show extracted data summary
    if [ -d "$mav0_dir" ]; then
        local gt_file="$mav0_dir/state_groundtruth_estimate0/groundtruth_normalized.txt"
        if [ -f "$gt_file" ]; then
            local gt_poses=$(wc -l < "$gt_file")
            log_info "📍 Ground truth poses: $gt_poses"
        fi
        
        local cam0_dir="$mav0_dir/cam0/data"
        if [ -d "$cam0_dir" ]; then
            local images=$(ls "$cam0_dir"/*.png 2>/dev/null | wc -l)
            log_info "📷 Stereo image pairs: $images"
        fi
    fi
    
    # Show SLAM results summary
    local orbslam_dir="$mav0_dir/output/orbslam3"
    local fs_dir1="$mav0_dir/output/foundation_stereo"
    local fs_dir2="$mav0_dir/output/foundationstereo"
    local fs_dir=""
    
    # Find the correct FoundationStereo directory
    if [ -d "$fs_dir1" ]; then
        fs_dir="$fs_dir1"
    elif [ -d "$fs_dir2" ]; then
        fs_dir="$fs_dir2"
    fi
    
    if [ -d "$orbslam_dir" ]; then
        local traj_file="$orbslam_dir/CameraTrajectory.txt"
        if [ -f "$traj_file" ]; then
            local poses=$(wc -l < "$traj_file")
            log_info "🔵 Regular ORB-SLAM3 poses: $poses"
        fi
    fi
    
    if [ -n "$fs_dir" ] && [ -d "$fs_dir" ]; then
        # Check for different possible trajectory file names
        local traj_file=""
        for possible_name in "CameraTrajectory_FoundationStereo.txt" "CameraTrajectory.txt" "trajectory.txt"; do
            if [ -f "$fs_dir/$possible_name" ]; then
                traj_file="$fs_dir/$possible_name"
                break
            fi
        done
        
        if [ -n "$traj_file" ]; then
            local poses=$(wc -l < "$traj_file")
            log_info "🔴 FoundationStereo poses: $poses"
        else
            log_info "🔴 FoundationStereo: output directory found but no trajectory file"
        fi
    fi
    
    # Show visualization results if available
    local viz_file="$output_dir/mav0_trajectory_comparison.png"
    if [ -f "$viz_file" ]; then
        log_info "\n📊 Trajectory Visualization:"
        log_success "  ✅ Comprehensive comparison generated: mav0_trajectory_comparison.png"
        
        # Extract visualization details from log if available
        local viz_log="$output_dir/logs/visualization.log"
        if [ -f "$viz_log" ]; then
            # Extract coordinate transformation info
            local transform=$(grep "Best transformation:" "$viz_log" 2>/dev/null | sed 's/.*Best transformation: //' | head -1)
            local score=$(grep "Combined score:" "$viz_log" 2>/dev/null | sed 's/.*Combined score: //' | head -1)
            
            if [ -n "$transform" ]; then
                log_info "  🔄 Coordinate transformation: $transform"
            fi
            if [ -n "$score" ]; then
                local percentage=$(echo "$score * 100" | bc 2>/dev/null || echo "N/A")
                if [ "$percentage" != "N/A" ]; then
                    log_info "  📈 Alignment quality: ${percentage}%"
                fi
            fi
        fi
    else
        log_warning "\n📊 Trajectory Visualization: Failed to generate"
    fi
    
    # Show output files
    log_info "\n📁 Generated files:"
    find "$output_dir" -name "*.png" -o -name "*.txt" -o -name "*.csv" | sort | while read -r file; do
        local rel_path="${file#$output_dir/}"
        log_info "  📄 $rel_path"
    done
    
    # Show logs location
    log_info "\n📋 Execution logs: $output_dir/logs/"
    
    log_success "\n🎉 Complete SLAM pipeline finished successfully!"
    if [ -f "$viz_file" ]; then
        log_info "📊 Open $output_dir/mav0_trajectory_comparison.png to view detailed trajectory comparison"
    fi
}

# Parse command line arguments
BAG_FILE=""
OUTPUT_DIR=""
SKIP_EXTRACT=false
REGULAR_ONLY=false
FS_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        -r|--regular-only)
            REGULAR_ONLY=true
            shift
            ;;
        -f|--fs-only)
            FS_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$BAG_FILE" ]; then
                BAG_FILE="$1"
            else
                log_error "Multiple bag files specified"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$BAG_FILE" ] && [ "$SKIP_EXTRACT" = false ]; then
    log_error "Bag file is required unless --skip-extract is used"
    show_help
    exit 1
fi

if [ "$SKIP_EXTRACT" = false ]; then
    check_file "$BAG_FILE"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi

if [ "$REGULAR_ONLY" = true ] && [ "$FS_ONLY" = true ]; then
    log_error "Cannot specify both --regular-only and --fs-only"
    exit 1
fi

# Create output directory and logs subdirectory
mkdir -p "$OUTPUT_DIR/logs"

# Show pipeline configuration
log_step "SLAM PIPELINE CONFIGURATION"
if [ "$SKIP_EXTRACT" = false ]; then
    log_info "Input bag file: $BAG_FILE"
fi
log_info "Output directory: $OUTPUT_DIR"
log_info "Skip extraction: $SKIP_EXTRACT"
log_info "Regular SLAM only: $REGULAR_ONLY"
log_info "FoundationStereo only: $FS_ONLY"
log_info "Verbose output: $VERBOSE"

# Main pipeline execution
main() {
    local start_time=$(date +%s)
    
    log_step "STARTING COMPLETE SLAM PIPELINE"
    log_info "Started at: $(date)"
    
    # Step 1: Extract ROS bag (unless skipped)
    if [ "$SKIP_EXTRACT" = false ]; then
        extract_bag_data "$BAG_FILE" "$OUTPUT_DIR"
    else
        log_step "STEP 1: Skipping ROS Bag Extraction"
        log_info "Using existing data in: $OUTPUT_DIR"
    fi
    
    local mav0_dir="$OUTPUT_DIR/mav0"
    check_directory "$mav0_dir"
    
    # Step 2: Synchronize timestamps and stereo pairs for ORB-SLAM3 compatibility
    synchronize_stereo_timestamps "$mav0_dir" "$OUTPUT_DIR"
    
    # Generate timestamp file
    generate_timestamps "$mav0_dir" "$OUTPUT_DIR"
    
    # Step 3: Run regular ORB-SLAM3 (unless FS only)
    if [ "$FS_ONLY" = false ]; then
        run_regular_slam "$mav0_dir" "$OUTPUT_DIR"
    fi
    
    # Step 4: Run FoundationStereo SLAM (unless regular only)
    if [ "$REGULAR_ONLY" = false ]; then
        run_foundationstereo_slam "$mav0_dir" "$OUTPUT_DIR"
    fi
    
    # Step 5: Generate visualization (don't fail pipeline if this fails)
    if generate_visualization "$mav0_dir" "$OUTPUT_DIR"; then
        log_success "Visualization generation completed successfully"
    else
        log_warning "Visualization generation failed, but SLAM results are still available"
        log_warning "You can manually run: python3 visualize_trajectories_directly.py $mav0_dir"
    fi
    
    # Show final summary
    show_summary "$OUTPUT_DIR"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    log_success "Pipeline completed in ${hours}h ${minutes}m ${seconds}s"
}

# Run main function
main "$@"
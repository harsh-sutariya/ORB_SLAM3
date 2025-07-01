# Complete SLAM Pipeline Automation

This repository provides a fully automated pipeline for processing ROS bag files through ORB-SLAM3 with both regular and FoundationStereo-enhanced processing.

## 🚀 Quick Start

### Basic Usage
```bash
# Process complete pipeline with both methods
./run_complete_slam_pipeline.sh /path/to/your/dataset.bag

# Use custom output directory
./run_complete_slam_pipeline.sh /path/to/dataset.bag -o my_analysis_results

# Run only FoundationStereo (faster for testing)
./run_complete_slam_pipeline.sh /path/to/dataset.bag -f

# Verbose output (see all processing details)
./run_complete_slam_pipeline.sh /path/to/dataset.bag -v
```

### Advanced Usage
```bash
# Skip extraction (use existing mav0 data)
./run_complete_slam_pipeline.sh /path/to/dataset.bag -s -o existing_results

# Run only regular ORB-SLAM3
./run_complete_slam_pipeline.sh /path/to/dataset.bag -r

# Get help with all options
./run_complete_slam_pipeline.sh --help
```

## 📊 Pipeline Steps

1. **Extract ROS Bag**: Converts ROS bag to mav0 format with normalized ground truth
2. **Regular ORB-SLAM3**: Processes dataset with standard ORB-SLAM3
3. **FoundationStereo SLAM**: Processes with deep learning enhanced stereo matching
4. **Visualization**: Creates comprehensive trajectory comparison plots

## 📁 Output Structure

After running the pipeline, you'll get:

```
slam_results_YYYYMMDD_HHMMSS/
├── mav0/                              # Extracted dataset
│   ├── cam0/data/                     # Left camera images
│   ├── cam1/data/                     # Right camera images
│   ├── imu0/data.csv                  # IMU measurements
│   ├── state_groundtruth_estimate0/
│   │   ├── groundtruth_normalized.txt # TUM format ground truth
│   │   └── data.csv                   # EuRoC format ground truth
│   └── output/
│       ├── orbslam3/                  # Regular SLAM results
│       │   ├── CameraTrajectory.txt
│       │   └── KeyFrameTrajectory.txt
│       └── foundation_stereo/          # FoundationStereo results
│           ├── CameraTrajectory_FoundationStereo.txt
│           └── KeyFrameTrajectory_FoundationStereo.txt
├── logs/                              # Execution logs
│   ├── extraction.log
│   ├── regular_slam.log
│   ├── foundationstereo_slam.log
│   └── visualization.log
└── mav0_trajectory_comparison.png     # Final comparison plot
```

## ⚙️ Requirements

- **ORB-SLAM3**: Built and ready to run
- **Python 3**: With matplotlib, pandas, numpy
- **Scripts**: `extract_bag_to_mav0.py` and `visualize_trajectories_directly.py`
- **Config**: `Examples/Stereo/Lunar.yaml` configuration file

## 🎯 Key Features

- **Fully Automated**: One command processes everything
- **Robust Error Handling**: Comprehensive logging and validation
- **Flexible Options**: Run individual components or full pipeline
- **Organized Outputs**: All results neatly structured
- **Progress Tracking**: Color-coded status messages
- **Timeout Protection**: Prevents infinite hanging

## 💡 Tips

- Use `-v` flag for debugging issues
- Check `logs/` directory for detailed error information
- Use `-s` flag to reprocess existing extractions
- Pipeline takes 15-30 minutes for typical datasets
- FoundationStereo is slower but provides better coverage

## 📋 Troubleshooting

**Common Issues:**
- **"Required file not found"**: Check ORB-SLAM3 build and paths
- **"Extraction failed"**: Verify ROS bag format and topics
- **"SLAM timeout"**: Normal for large datasets, check logs for progress
- **"Visualization failed"**: Ensure Python packages are installed

**Log Locations:**
- Extraction: `output_dir/logs/extraction.log`
- Regular SLAM: `output_dir/logs/regular_slam.log`
- FoundationStereo: `output_dir/logs/foundationstereo_slam.log`
- Visualization: `output_dir/logs/visualization.log` 
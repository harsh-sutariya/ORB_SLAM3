# FoundationStereo Integration with ORB-SLAM3

This document describes the integration of the FoundationStereo deep learning model for stereo disparity computation within the ORB-SLAM3 Frame class.

## Overview

The `Frame::GetStereoDisparity()` function provides a way to compute stereo disparity using the FoundationStereo deep learning model, which can potentially provide more accurate disparity maps compared to traditional stereo matching methods.

## Prerequisites

1. **FoundationStereo Model**: You need to have the FoundationStereo model and its dependencies set up
2. **Python Environment**: Python with required packages (torch, opencv-python, numpy, etc.)
3. **Pretrained Model**: The pretrained model checkpoint at `./pretrained_models/23-51-11/model_best_bp2.pth`
4. **Modified Python Script**: The `scripts/run_demo_for_cpp.py` script (included in this integration)

## Files Modified/Added

### Modified Files:
- `src/Frame.cc` - Added `GetStereoDisparity()` function implementation
- `include/Frame.h` - Added function declaration

### New Files:
- `scripts/run_demo_for_cpp.py` - Modified Python script that saves disparity in C++ readable formats
- `Examples/test_stereo_disparity.cpp` - Example usage program
- `Examples/CMakeLists_stereo_addition.txt` - CMake configuration for the example

## Function Signature

```cpp
cv::Mat Frame::GetStereoDisparity(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                  const std::string &outputDir = "./test_outputs/");
```

### Parameters:
- `imLeft`: Left stereo image (cv::Mat)
- `imRight`: Right stereo image (cv::Mat) 
- `outputDir`: Output directory for temporary files (default: "./test_outputs/")

### Returns:
- `cv::Mat`: Disparity map as floating-point matrix (CV_32F)
- Empty `cv::Mat` if computation fails

## Usage Example

```cpp
#include "Frame.h"

// Load stereo images
cv::Mat imgLeft = cv::imread("left.png");
cv::Mat imgRight = cv::imread("right.png");

// Create Frame object (you'll need proper ORB extractors, vocabulary, etc.)
Frame frame(imgLeft, imgRight, timestamp, extractorLeft, extractorRight, 
            voc, K, distCoef, bf, thDepth, pCamera);

// Compute disparity using FoundationStereo
cv::Mat disparity = frame.GetStereoDisparity(imgLeft, imgRight, "./output/");

if (!disparity.empty()) {
    std::cout << "Disparity computation successful!" << std::endl;
    std::cout << "Disparity size: " << disparity.size() << std::endl;
}
```

## Build Instructions

1. Add the CMake configuration to your main CMakeLists.txt:
```cmake
# Include the example build configuration
include(Examples/CMakeLists_stereo_addition.txt)
```

2. Build the project:
```bash
mkdir build && cd build
cmake ..
make -j4
```

3. Build the test example:
```bash
make test_stereo_disparity
```

## Running the Example

```bash
# From the ORB_SLAM3 root directory
./Examples/test_stereo_disparity path/to/left.png path/to/right.png
```

## Directory Structure Requirements

Make sure your directory structure looks like this:
```
ORB_SLAM3/
├── scripts/
│   ├── run_demo.py (original)
│   └── run_demo_for_cpp.py (modified for C++)
├── pretrained_models/
│   └── 23-51-11/
│       ├── model_best_bp2.pth
│       └── cfg.yaml
├── test_outputs/ (will be created automatically)
└── ... (other ORB-SLAM3 files)
```

## Notes and Limitations

1. **Performance**: The deep learning inference will be slower than traditional stereo matching
2. **GPU Requirements**: Requires CUDA-capable GPU for reasonable performance
3. **Dependencies**: Requires Python environment with PyTorch, OpenCV, and other dependencies
4. **Temporary Files**: Function creates temporary files but cleans them up automatically
5. **Error Handling**: Returns empty cv::Mat on failure - check return value

## Integration with ORB-SLAM3 Pipeline

You can integrate this function into the ORB-SLAM3 pipeline by:

1. **Replacing traditional stereo matching**: Replace calls to `ComputeStereoMatches()` with `GetStereoDisparity()`
2. **Preprocessing disparity**: Convert the dense disparity map to sparse keypoint depths as needed
3. **Caching results**: Consider caching disparity results to avoid recomputation

## Troubleshooting

### Common Issues:

1. **Python script not found**: Ensure `scripts/run_demo_for_cpp.py` exists and is executable
2. **Model not found**: Check that the pretrained model path is correct
3. **CUDA errors**: Ensure your system has compatible CUDA installation
4. **Permission errors**: Ensure write permissions for the output directory

### Debug Tips:

- Check console output for detailed error messages
- Verify temporary files are created in the output directory
- Test the Python script independently before using from C++

## Future Improvements

1. **Direct PyTorch Integration**: Embed the model directly in C++ using LibTorch
2. **Asynchronous Processing**: Run disparity computation in background thread
3. **Model Optimization**: Use TensorRT or similar for faster inference
4. **Memory Management**: Improve memory efficiency for large images

## Command Used

The function essentially executes this Python command:
```bash
python scripts/run_demo_for_cpp.py \
    --left_file /path/to/left.png \
    --right_file /path/to/right.png \
    --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
    --out_dir /path/to/output \
    --get_pc 0 \
    --scale 1.0 \
    --valid_iters 32
``` 
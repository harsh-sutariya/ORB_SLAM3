/**
* Example usage of Frame::GetStereoDisparity function
* This demonstrates how to use the deep learning-based stereo disparity computation
* integrated into the ORB-SLAM3 Frame class.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "../include/Frame.h"
#include "../include/ORBextractor.h"
#include "../include/ORBVocabulary.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "Usage: ./test_stereo_disparity <left_image> <right_image>" << endl;
        return -1;
    }

    // Load stereo images
    string leftImagePath = argv[1];
    string rightImagePath = argv[2];
    
    Mat imgLeft = imread(leftImagePath, IMREAD_COLOR);
    Mat imgRight = imread(rightImagePath, IMREAD_COLOR);
    
    if (imgLeft.empty() || imgRight.empty()) {
        cerr << "Error: Could not load images!" << endl;
        return -1;
    }
    
    cout << "Loaded images: " << leftImagePath << " and " << rightImagePath << endl;
    cout << "Left image size: " << imgLeft.size() << endl;
    cout << "Right image size: " << imgRight.size() << endl;
    
    // Create dummy ORB extractor and vocabulary (not used for disparity computation)
    // These are needed for Frame constructor but won't be used in GetStereoDisparity
    ORBextractor* extractorLeft = new ORBextractor(1000, 1.2f, 8, 20, 7);
    ORBextractor* extractorRight = new ORBextractor(1000, 1.2f, 8, 20, 7);
    ORBVocabulary* voc = new ORBVocabulary();
    
    // Camera calibration matrix (example values - adjust for your camera)
    Mat K = (Mat_<float>(3,3) << 
        718.856, 0, 607.1928,
        0, 718.856, 185.2157,
        0, 0, 1);
    
    Mat distCoef = Mat::zeros(5, 1, CV_32F);  // No distortion
    
    float bf = 387.5718;  // baseline * fx (example value)
    float thDepth = 40.0; // depth threshold
    
    // Create a simple pinhole camera model (you might need to adjust this)
    GeometricCamera* pCamera = nullptr; // You'll need to create proper camera object
    
    // Create Frame object
    double timestamp = 0.0;
    IMU::Calib imuCalib; // Default IMU calibration
    
    // Note: For this example, we'll create a simplified frame constructor call
    // In practice, you'll use the appropriate constructor based on your setup
    Frame frame(imgLeft, imgRight, timestamp, extractorLeft, extractorRight, voc, 
                K, distCoef, bf, thDepth, pCamera, nullptr, imuCalib);
    
    cout << "Frame created with ID: " << frame.mnId << endl;
    
    // Call the GetStereoDisparity function - this now calls run_demo.py directly
    string outputDir = "./test_outputs";
    Mat disparity = frame.GetStereoDisparity(imgLeft, imgRight, outputDir);
    
    if (disparity.empty()) {
        cerr << "Error: Disparity computation failed!" << endl;
        return -1;
    }
    
    cout << "Disparity computation successful!" << endl;
    cout << "Disparity map size: " << disparity.size() << endl;
    cout << "Disparity map type: " << disparity.type() << endl;
    
    // Find min and max disparity values for analysis
    double minDisp, maxDisp;
    minMaxLoc(disparity, &minDisp, &maxDisp);
    cout << "Disparity range: [" << minDisp << ", " << maxDisp << "]" << endl;
    
    cout << "Raw FoundationStereo output saved to " << outputDir << endl;
    cout << "Files generated:" << endl;
    cout << "  - vis.png (visualization from FoundationStereo)" << endl;
    cout << "  - depth_meter.npy (depth in meters)" << endl;
    cout << "  - cloud.ply (point cloud)" << endl;
    cout << "  - cloud_denoise.ply (denoised point cloud)" << endl;
    
    // Load and display the original FoundationStereo visualization
    Mat originalVis = imread(outputDir + "/vis.png");
    if (!originalVis.empty()) {
        namedWindow("FoundationStereo Output", WINDOW_AUTOSIZE);
        imshow("FoundationStereo Output", originalVis);
    } else {
        cout << "Warning: Could not load FoundationStereo visualization" << endl;
    }
    
    cout << "Press any key to exit..." << endl;
    waitKey(0);
    
    // Cleanup
    delete extractorLeft;
    delete extractorRight;
    delete voc;
    
    return 0;
} 
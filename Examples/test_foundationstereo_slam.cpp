/**
* Practical example of using FoundationStereo as stereo backend in ORB-SLAM3
* This demonstrates different integration strategies for real SLAM systems.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "../include/Frame.h"
#include "../include/ORBextractor.h"
#include "../include/ORBVocabulary.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

int main(int argc, char** argv)
{
    if (argc != 4) {
        cout << "Usage: ./test_foundationstereo_slam <left_image> <right_image> <method>" << endl;
        cout << "Methods: traditional, foundationstereo, hybrid" << endl;
        return -1;
    }

    string leftImagePath = argv[1];
    string rightImagePath = argv[2];
    string method = argv[3];
    
    // Load stereo images
    Mat imgLeft = imread(leftImagePath, IMREAD_COLOR);
    Mat imgRight = imread(rightImagePath, IMREAD_COLOR);
    
    if (imgLeft.empty() || imgRight.empty()) {
        cerr << "Error: Could not load images!" << endl;
        return -1;
    }
    
    cout << "=== FoundationStereo SLAM Integration Demo ===" << endl;
    cout << "Left image: " << leftImagePath << endl;
    cout << "Right image: " << rightImagePath << endl;
    cout << "Method: " << method << endl;
    cout << "Image size: " << imgLeft.size() << endl << endl;
    
    // Setup ORB-SLAM3 components
    ORBextractor* extractorLeft = new ORBextractor(1000, 1.2f, 8, 20, 7);
    ORBextractor* extractorRight = new ORBextractor(1000, 1.2f, 8, 20, 7);
    ORBVocabulary* voc = new ORBVocabulary();
    
    // EuRoC camera parameters
    Mat K = (Mat_<float>(3,3) << 
        458.654, 0, 367.215,
        0, 457.296, 248.375,
        0, 0, 1);
    
    Mat distCoef = Mat::zeros(5, 1, CV_32F);
    float bf = 458.654 * 0.110; // fx * baseline
    float thDepth = 40.0;
    
    GeometricCamera* pCamera = nullptr;
    double timestamp = 0.0;
    IMU::Calib imuCalib;
    
    cout << "Creating Frame object..." << endl;
    Frame frame(imgLeft, imgRight, timestamp, extractorLeft, extractorRight, voc, 
                K, distCoef, bf, thDepth, pCamera, nullptr, imuCalib);
    
    cout << "Frame created with ID: " << frame.mnId << endl;
    cout << "ORB keypoints extracted: " << frame.N << endl << endl;
    
    // Time the stereo matching process
    auto start = chrono::high_resolution_clock::now();
    
    if (method == "traditional") {
        cout << "=== Using Traditional Stereo Matching ===" << endl;
        frame.ComputeStereoMatches();
        
    } else if (method == "foundationstereo") {
        cout << "=== Using Pure FoundationStereo ===" << endl;
        frame.ComputeStereoMatchesFoundationStereo(imgLeft, imgRight, "./foundationstereo_output");
        
    } else if (method == "hybrid") {
        cout << "=== Using Hybrid Approach (Traditional + FoundationStereo) ===" << endl;
        frame.ComputeStereoMatchesHybrid(imgLeft, imgRight, "./hybrid_output");
        
    } else {
        cerr << "Unknown method: " << method << endl;
        return -1;
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "\n=== Stereo Matching Results ===" << endl;
    cout << "Processing time: " << duration.count() << " ms" << endl;
    
    // Analyze results
    int validDepths = 0;
    float minDepth = FLT_MAX, maxDepth = 0;
    float totalDepth = 0;
    
    for(int i = 0; i < frame.N; i++) {
        if(frame.mvDepth[i] > 0) {
            validDepths++;
            minDepth = min(minDepth, frame.mvDepth[i]);
            maxDepth = max(maxDepth, frame.mvDepth[i]);
            totalDepth += frame.mvDepth[i];
        }
    }
    
    cout << "Valid depth measurements: " << validDepths << " / " << frame.N 
         << " (" << (100.0f * validDepths / frame.N) << "%)" << endl;
    
    if(validDepths > 0) {
        float meanDepth = totalDepth / validDepths;
        cout << "Depth range: [" << minDepth << ", " << maxDepth << "] meters" << endl;
        cout << "Mean depth: " << meanDepth << " meters" << endl;
        
        // Show some example 3D points
        cout << "\nExample 3D points (first 5 valid):" << endl;
        int shown = 0;
        for(int i = 0; i < frame.N && shown < 5; i++) {
            if(frame.mvDepth[i] > 0) {
                Eigen::Vector3f x3D;
                if(frame.UnprojectStereo(i, x3D)) {
                    cout << "  Point " << i << ": (" << x3D.x() << ", " << x3D.y() << ", " << x3D.z() << ")" << endl;
                    shown++;
                }
            }
        }
    }
    
    cout << "\n=== Integration Analysis ===" << endl;
    
    if (method == "traditional") {
        cout << "Traditional stereo matching:" << endl;
        cout << "✓ Fast processing (~" << duration.count() << " ms)" << endl;
        cout << "✓ Low memory usage" << endl;
        cout << "? Coverage depends on image texture" << endl;
        cout << "? May fail in low-texture regions" << endl;
        
    } else if (method == "foundationstereo") {
        cout << "FoundationStereo deep learning:" << endl;
        cout << "✓ Dense disparity computation" << endl;
        cout << "✓ Works in challenging scenarios" << endl;
        cout << "✓ High accuracy" << endl;
        cout << "- Slower processing (~" << duration.count() << " ms)" << endl;
        cout << "- Requires GPU for real-time use" << endl;
        
    } else if (method == "hybrid") {
        cout << "Hybrid approach:" << endl;
        cout << "✓ Best of both worlds" << endl;
        cout << "✓ Fallback for low-coverage scenarios" << endl;
        cout << "✓ Adaptive performance" << endl;
        cout << "- Processing time varies with coverage" << endl;
    }
    
    cout << "\n=== SLAM Pipeline Ready! ===" << endl;
    cout << "The Frame object now has populated mvDepth[] and mvuRight[] vectors." << endl;
    cout << "You can now use this frame in the standard ORB-SLAM3 pipeline:" << endl;
    cout << "- Tracking & Localization" << endl;
    cout << "- Map Point Creation" << endl;
    cout << "- Loop Closure Detection" << endl;
    cout << "- Bundle Adjustment" << endl;
    cout << "- Dense Mapping (if desired)" << endl;
    
    // Cleanup
    delete extractorLeft;
    delete extractorRight;
    delete voc;
    
    return 0;
} 
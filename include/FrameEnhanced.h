/**
* Enhanced Frame class for ORB-SLAM3 with FoundationStereo integration
*
* This class extends the original Frame functionality with deep learning-based
* stereo disparity computation using FoundationStereo model.
*/

#ifndef FRAME_ENHANCED_H
#define FRAME_ENHANCED_H

#include "Frame.h"
#include <string>

namespace ORB_SLAM3
{

class FrameEnhanced : public Frame
{
public:
    // Inherit constructors from base Frame class
    using Frame::Frame;
    
    // Enhanced stereo matching that can use FoundationStereo
    void ComputeStereoMatchesEnhanced(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                    bool useFoundationStereo = false, 
                                    const std::string &outputDir = "./test_outputs");
    
    // Combine traditional stereo matching with FoundationStereo disparity
    void CombineStereoDisparities(const cv::Mat &traditionalDisparity, 
                                const cv::Mat &foundationStereoDisparity,
                                float alpha = 0.3f); // Weight for FoundationStereo (0.0 = traditional only, 1.0 = FoundationStereo only)
    
    // Validate and refine disparity using FoundationStereo
    void RefineDepthWithFoundationStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                       const std::string &outputDir = "./test_outputs");
    
    // Get enhanced depth map combining ORB features with FoundationStereo
    cv::Mat GetEnhancedDepthMap(const cv::Mat &imLeft, const cv::Mat &imRight,
                              const std::string &outputDir = "./test_outputs");
    
    // Statistics for analysis
    struct EnhancedStereoStats {
        int traditionalMatches = 0;
        int foundationStereoPixels = 0;
        int combinedMatches = 0;
        double meanTraditionalDisparity = 0.0;
        double meanFoundationStereoDisparity = 0.0;
        double disparityAgreement = 0.0; // Percentage of pixels where both methods agree
        double computationTime = 0.0;
    };
    
    EnhancedStereoStats mEnhancedStats;
    
    // Configuration options
    struct FoundationStereoConfig {
        bool enabled = false;
        float weight = 0.3f; // How much to weight FoundationStereo vs traditional
        int skipFrames = 10; // Only compute FoundationStereo every N frames
        float disparityThreshold = 2.0f; // Max disparity difference to consider agreement
        std::string modelPath = "./pretrained_models/23-51-11/model_best_bp2.pth";
        std::string outputDir = "./test_outputs";
        bool saveIntermediateResults = false;
    };
    
    static FoundationStereoConfig sFoundationStereoConfig;
    
    // Enable/disable FoundationStereo integration
    static void EnableFoundationStereo(bool enable = true, 
                                     float weight = 0.3f, 
                                     int skipFrames = 10);
    
    // Set FoundationStereo configuration
    static void SetFoundationStereoConfig(const FoundationStereoConfig &config);

private:
    // Helper function to convert disparity to depth
    void DisparityToDepth(const cv::Mat &disparity, std::vector<float> &depths, 
                         const std::vector<cv::KeyPoint> &keypoints);
    
    // Helper function to validate disparity consistency
    bool ValidateDisparityConsistency(const cv::Mat &leftDisparity, 
                                    const cv::Mat &rightDisparity,
                                    float threshold = 1.0f);
    
    // Cache for FoundationStereo results to avoid recomputation
    static std::map<std::string, cv::Mat> sDisparityCache;
    static int sCacheMaxSize;
    
    // Frame counter for skipping frames
    static int sFrameCounter;
};

} // namespace ORB_SLAM3

#endif // FRAME_ENHANCED_H 
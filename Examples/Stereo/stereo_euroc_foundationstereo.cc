/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{  
    if(argc < 5)
    {
        cerr << endl << "Usage: ./stereo_euroc_foundationstereo path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)" << endl;
        cerr << endl << "This example integrates FoundationStereo deep learning model for enhanced stereo disparity computation." << endl;
        return 1;
    }

    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeft;
    vector< vector<string> > vstrImageRight;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;

    vstrImageLeft.resize(num_seq);
    vstrImageRight.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(2*seq) + 3]);
        string pathTimeStamps(argv[(2*seq) + 4]);

        string pathCam0 = pathSeq + "/mav0/cam0/data";
        string pathCam1 = pathSeq + "/mav0/cam1/data";

        LoadImages(pathCam0, pathCam1, pathTimeStamps, vstrImageLeft[seq], vstrImageRight[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vector<float> vTimesFoundationStereo;
    vTimesTrack.resize(tot_images);
    vTimesFoundationStereo.resize(tot_images);

    cout << endl << "-------" << endl;
    cout << "Starting ORB-SLAM3 with FoundationStereo integration" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::STEREO, true);

    cv::Mat imLeft, imRight;
    int image_count = 0;
    
    for (seq = 0; seq<num_seq; seq++)
    {
        // Seq loop
        double t_resize = 0;
        double t_rect = 0;
        double t_track = 0;
        double t_foundationstereo = 0;
        int num_rect = 0;
        int proccIm = 0;
        
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++, image_count++)
        {
            // Read left and right images from file
            imLeft = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_UNCHANGED);
            imRight = cv::imread(vstrImageRight[seq][ni],cv::IMREAD_UNCHANGED);

            if(imLeft.empty())
            {
                cerr << endl << "Failed to load image at: "
                     << string(vstrImageLeft[seq][ni]) << endl;
                return 1;
            }

            if(imRight.empty())
            {
                cerr << endl << "Failed to load image at: "
                     << string(vstrImageRight[seq][ni]) << endl;
                return 1;
            }

            double tframe = vTimestampsCam[seq][ni];

            // Optional: Compute FoundationStereo disparity for enhanced depth estimation
            cv::Mat foundationStereoDisparity;
            #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_fs_start = std::chrono::steady_clock::now();
            #else
                std::chrono::monotonic_clock::time_point t_fs_start = std::chrono::monotonic_clock::now();
            #endif
            
            // Create a dummy frame to access GetStereoDisparity function
            // This is a demonstration - in a real implementation you'd integrate this directly into Frame constructor
            if (ni % 10 == 0) // Only compute FoundationStereo every 10 frames to save computation
            {
                cout << "Computing FoundationStereo disparity for frame " << ni << " of sequence " << seq << endl;
                
                // Create output directory
                string outputDir = "./test_outputs/foundationstereo_sequence_" + to_string(seq);
                string mkdirCmd = "mkdir -p \"" + outputDir + "\"";
                system(mkdirCmd.c_str());
                
                // Use a Frame instance to compute FoundationStereo disparity
                // Note: This creates a temporary frame just for disparity computation
                ORB_SLAM3::Frame tempFrame;
                foundationStereoDisparity = tempFrame.GetStereoDisparity(imLeft, imRight, outputDir);
                
                if (!foundationStereoDisparity.empty()) {
                    cout << "FoundationStereo disparity computed successfully. Size: " 
                         << foundationStereoDisparity.cols << "x" << foundationStereoDisparity.rows << endl;
                    
                    // Save disparity for analysis
                    string disparityPath = outputDir + "/frame_" + to_string(ni) + "_disparity.png";
                    cv::imwrite(disparityPath, foundationStereoDisparity);
                    
                    // Display statistics
                    double minVal, maxVal;
                    cv::minMaxLoc(foundationStereoDisparity, &minVal, &maxVal);
                    cv::Scalar meanVal = cv::mean(foundationStereoDisparity);
                    cout << "Disparity stats - Min: " << minVal << ", Max: " << maxVal 
                         << ", Mean: " << meanVal[0] << endl;
                }
            }
            
            #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_fs_end = std::chrono::steady_clock::now();
            #else
                std::chrono::monotonic_clock::time_point t_fs_end = std::chrono::monotonic_clock::now();
            #endif
            
            t_foundationstereo = std::chrono::duration_cast<std::chrono::duration<double> >(t_fs_end - t_fs_start).count();
            vTimesFoundationStereo[image_count] = t_foundationstereo;

            #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            #else
                std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
            #endif

            // Pass the images to the SLAM system
            SLAM.TrackStereo(imLeft,imRight,tframe, vector<ORB_SLAM3::IMU::Point>(), vstrImageLeft[seq][ni]);

            #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            #else
                std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
            #endif

#ifdef REGISTER_TIMES
            t_track = t_resize + t_rect + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[image_count]=ttrack;
            
            cout << "Frame " << ni << " - Track time: " << ttrack << "s";
            if (t_foundationstereo > 0) {
                cout << ", FoundationStereo time: " << t_foundationstereo << "s";
            }
            cout << endl;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6);
        }

        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }

    }
    
    // Stop all threads
    SLAM.Shutdown();

    // Timing Statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<tot_images; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[tot_images/2] << endl;
    cout << "mean tracking time: " << totaltime/tot_images << endl;
    
    // FoundationStereo timing statistics
    float totalFoundationStereoTime = 0;
    int foundationStereoFrames = 0;
    for(int ni=0; ni<tot_images; ni++)
    {
        if(vTimesFoundationStereo[ni] > 0) {
            totalFoundationStereoTime += vTimesFoundationStereo[ni];
            foundationStereoFrames++;
        }
    }
    
    if(foundationStereoFrames > 0) {
        cout << "FoundationStereo frames processed: " << foundationStereoFrames << endl;
        cout << "Mean FoundationStereo time: " << totalFoundationStereoTime/foundationStereoFrames << endl;
    }

    // Save camera trajectory
    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + "_foundationstereo.txt";
        const string f_file =  "f_" + string(argv[argc-1]) + "_foundationstereo.txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory_FoundationStereo.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory_FoundationStereo.txt");
    }

    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);
        }
    }
} 
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
#include<sys/stat.h>
#include <dirent.h> // For directory listing
#include <signal.h>
#include <atomic>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

std::atomic<bool> keep_running(true);

void sigint_handler(int) {
    keep_running = false;
}

// Helper to get sorted list of files in a directory with a given prefix
std::vector<std::string> GetSortedFiles(const std::string& dir, const std::string& prefix) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) return files;
    struct dirent* ep;
    while ((ep = readdir(dp))) {
        std::string fname = ep->d_name;
        if (fname.find(prefix) == 0 && fname.substr(fname.size()-4) == ".png") {
            files.push_back(dir + "/" + fname);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

// Helper function to create output directory structure
string CreateOutputDirectory(const string &pathSeq, const string &method)
{
    string outputDir = pathSeq + "/mav0/output/" + method;
    
    // Create directory structure
    string mkdirCmd = "mkdir -p \"" + outputDir + "\"";
    int result = system(mkdirCmd.c_str());
    
    if (result == 0) {
        cout << "Created output directory: " << outputDir << endl;
    } else {
        cout << "Warning: Could not create directory " << outputDir << ". Files will be saved to current directory." << endl;
        return "";
    }
    
    return outputDir;
}

int main(int argc, char **argv)
{  
    if(argc < 4)
    {
        cerr << endl << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings path_to_sequence_folder" << endl;

        return 1;
    }

    signal(SIGINT, sigint_handler);

    string vocab_path(argv[1]);
    string settings_path(argv[2]);
    string seq_path(argv[3]);

    string pathCam0 = seq_path + "/mav0/cam0/data";
    string pathCam1 = seq_path + "/mav0/cam1/data";

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(vocab_path, settings_path, ORB_SLAM3::System::STEREO, true);

    size_t last_processed = 0;
    double dt = 0.1; // 10Hz, adjust as needed
    cout << "[INFO] Starting incremental SLAM. Press Ctrl+C to exit." << endl;

    while (keep_running) {
        // Get sorted left/right image lists
        std::vector<std::string> left_images = GetSortedFiles(pathCam0, "left_");
        std::vector<std::string> right_images = GetSortedFiles(pathCam1, "right_");
        size_t n = std::min(left_images.size(), right_images.size());

        // Process any new pairs
        for (size_t i = last_processed; i < n; ++i) {
            cv::Mat imLeft = cv::imread(left_images[i], cv::IMREAD_UNCHANGED);
            cv::Mat imRight = cv::imread(right_images[i], cv::IMREAD_UNCHANGED);
            if (imLeft.empty() || imRight.empty()) {
                cerr << "[WARN] Failed to load image pair: " << left_images[i] << ", " << right_images[i] << endl;
                continue;
            }
            double tframe = i * dt;
            cout << "[INFO] Processing pair " << i << ": " << left_images[i] << ", " << right_images[i] << endl;
            SLAM.TrackStereo(imLeft, imRight, tframe, std::vector<ORB_SLAM3::IMU::Point>(), left_images[i]);
        }
        last_processed = n;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Sleep before checking for new images
    }

    cout << "[INFO] Shutting down SLAM system..." << endl;
    SLAM.Shutdown();
    cout << "[INFO] Done." << endl;
    return 0;
}

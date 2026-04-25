// KCF Video Tracker Demo - OpenCV 4.x compatible
// Reads from video file instead of image sequence

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "kcftracker.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    cout << "KCF Video Tracker (RISC-V target)\n";

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_path> [output_path]\n";
        cerr << "Options: hog, fixed_window, singlescale, lab, gray, show\n";
        return 1;
    }

    // Parse options
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = false;

    for (int i = 2; i < argc; i++) {
        string opt = argv[i];
        if (opt == "hog") HOG = true;
        if (opt == "fixed_window") FIXEDWINDOW = true;
        if (opt == "singlescale") MULTISCALE = false;
        if (opt == "show") SILENT = false;
        if (opt == "lab") { LAB = true; HOG = true; }
        if (opt == "gray") HOG = false;
    }

    string videoPath = argv[1];
    string outputPath = (argc > 2 && argv[argc-1][0] != '-') ? argv[argc-1] : "output.txt";

    // Open video
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video " << videoPath << "\n";
        return 1;
    }

    // Read first frame
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: Empty video\n";
        return 1;
    }

    // Default ROI (center region, can be overridden by region.txt)
    float xMin = frame.cols / 4.0;
    float yMin = frame.rows / 4.0;
    float width = frame.cols / 2.0;
    float height = frame.rows / 2.0;

    // Try to read region.txt for groundtruth
    ifstream gtFile("region.txt");
    if (gtFile.is_open()) {
        string line;
        getline(gtFile, line);
        gtFile.close();

        // Parse: x1,y1,x2,y2,x3,y3,x4,y4
        float x1, y1, x2, y2, x3, y3, x4, y4;
        char ch;
        istringstream ss(line);
        ss >> x1 >> ch >> y1 >> ch >> x2 >> ch >> y2 >> ch >> x3 >> ch >> y3 >> ch >> x4 >> ch >> y4;

        xMin = min(x1, min(x2, min(x3, x4)));
        yMin = min(y1, min(y2, min(y3, y4)));
        width = max(x1, max(x2, max(x3, x4))) - xMin;
        height = max(y1, max(y2, max(y3, y4))) - yMin;
    }

    cout << "Video: " << frame.cols << "x" << frame.rows << "\n";
    cout << "Initial ROI: [" << xMin << ", " << yMin << ", " << width << ", " << height << "]\n";
    cout << "Features: HOG=" << HOG << ", LAB=" << LAB << ", Multiscale=" << MULTISCALE << "\n";

    // Create KCFTracker
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    tracker.init(Rect(xMin, yMin, width, height), frame);

    // Process video
    int nFrames = 0;
    int trackSuccess = 0;
    ofstream resultsFile(outputPath);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        nFrames++;
        Rect result = tracker.update(frame);

        // Write result
        resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << "\n";

        // Draw rectangle
        rectangle(frame, Point(result.x, result.y),
                  Point(result.x + result.width, result.y + result.height),
                  Scalar(0, 255, 255), 2);

        // Display if not silent
        if (!SILENT) {
            imshow("KCF Tracker", frame);
            waitKey(1);
        }

        // Count successful tracking (result area > 0)
        if (result.width > 0 && result.height > 0) trackSuccess++;
    }

    cap.release();
    resultsFile.close();

    cout << "Tracking finished.\n";
    cout << "Frames: " << nFrames << "\n";
    cout << "Success rate: " << (100.0 * trackSuccess / nFrames) << "%\n";
    cout << "Results saved to: " << outputPath << "\n";

    return 0;
}
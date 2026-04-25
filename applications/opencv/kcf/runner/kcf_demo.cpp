#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

// KCF Tracker demo for RISC-V RVV profiling
// Targets: FFT, HOG features, Gaussian kernel, correlation

int main(int argc, char* argv[]) {
    std::cout << "OpenCV KCF Tracker Demo (RISC-V RVV target)\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [output_path]\n";
        return 1;
    }

    const std::string videoPath = argv[1];
    const std::string outputPath = (argc > 2) ? argv[2] : "output.mp4";

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video " << videoPath << "\n";
        return 1;
    }

    // Read first frame for ROI selection
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Empty video\n";
        return 1;
    }

    // Default bounding box (center region)
    int centerX = frame.cols / 2;
    int centerY = frame.rows / 2;
    int boxWidth = std::min(100, frame.cols / 4);
    int boxHeight = std::min(100, frame.rows / 4);
    cv::Rect2d roi(centerX - boxWidth/2, centerY - boxHeight/2, boxWidth, boxHeight);

    std::cout << "Initial ROI: " << roi << "\n";
    std::cout << "Video size: " << frame.cols << "x" << frame.rows << "\n";

    // Create KCF tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    tracker->init(frame, roi);

    std::cout << "Tracking started...\n";

    // Process frames
    int frameCount = 0;
    int trackSuccess = 0;
    cv::VideoWriter writer;

    while (cap >> frame) {
        if (frame.empty()) break;

        frameCount++;
        bool success = tracker->update(frame, roi);

        if (success) {
            trackSuccess++;
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::rectangle(frame, roi, cv::Scalar(0, 0, 255), 2);
        }

        // Draw frame counter
        cv::putText(frame, std::to_string(frameCount),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 255, 255), 2);

        // Write output (optional)
        if (!writer.isOpened() && argc > 2) {
            writer.open(outputPath,
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        cap.get(cv::CAP_PROP_FPS),
                        frame.size());
        }
        if (writer.isOpened()) {
            writer << frame;
        }
    }

    std::cout << "Tracking finished.\n";
    std::cout << "Frames processed: " << frameCount << "\n";
    std::cout << "Tracking success rate: "
              << (100.0 * trackSuccess / frameCount) << "%\n";

    return 0;
}
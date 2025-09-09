#include <my_webcam.hpp>

MyWebcam::MyWebcam(const std::string camName, const std::string deviceName, int frameWidth, int frameHeight, int FPS)
    : camName_(camName), deviceName_(deviceName), frameWidth_(frameWidth), frameHeight_(frameHeight), FPS_(FPS) {
    // Open camera with V4L2 backend
    cap_.open(deviceName_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        throw std::runtime_error("Error: Could not open video device " + deviceName_);
    } else {
        std::cout << "Successfully opened video device " << deviceName_
            << " for camera " << camName_ << std::endl;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight_);
    cap_.set(cv::CAP_PROP_FPS, FPS_); 
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
}

MyWebcam::~MyWebcam() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

int MyWebcam::readFrame(cv::Mat& frame, std::string& errMsg) {
    // Check if camera is opened
    if (!cap_.isOpened()) {
        errMsg = "Error: Video device " + deviceName_ + " is not opened.";
        return -1;
    }
    // Capture frame
    if (!cap_.read(frame)) {
        errMsg = "Error: Could not read frame from " + camName_;
        return -1;
    }
    // Check valid frame
    if (frame.empty()) {
        errMsg = "Error: Frame is empty from " + camName_;
        return -1;
    } 
    return 0;
}
#include <my_webcam.h>

MyWebcam::MyWebcam(const std::string cam_name, const std::string device_name, int frame_width, int frame_height)
    : cam_name_(cam_name), device_name_(device_name), frame_width_(frame_width), frame_height_(frame_height) {
    // Open camera with V4L2 backend
    cap.open(device_name_, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video device " + device_name_);
    } else {
        std::cout << "Successfully opened video device " << device_name_ 
            << " for camera " << cam_name_ << std::endl;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
}

MyWebcam::~MyWebcam() {
    if (cap.isOpened()) {
        cap.release();
    }
}

int MyWebcam::ReadFrame(cv::Mat& frame, std::string& err_msg) {
    if (!cap.isOpened()) {
        err_msg = "Error: Video device " + device_name_ + " is not opened.";
        return -1;
    }

    if (!cap.read(frame)) {
        err_msg = "Error: Could not read frame from " + cam_name_;
        return -1;
    }

    if (frame.empty()) {
        err_msg = "Error: Frame is empty from " + cam_name_;
        return -1;
    } 

    return 0;
}
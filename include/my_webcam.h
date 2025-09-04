#ifndef MY_WEBCAM_H
#define MY_WEBCAM_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <iostream>

class MyWebcam
{
public:
    MyWebcam(const std::string cam_name, const std::string device_name, int frame_width, int frame_height);
    ~MyWebcam();
    int ReadFrame(cv::Mat& frame, std::string& err_msg);

private:
    cv::VideoCapture cap;
    std::string cam_name_;
    std::string device_name_;
    int frame_width_;
    int frame_height_;
};

#endif // MY_WEBCAM_H
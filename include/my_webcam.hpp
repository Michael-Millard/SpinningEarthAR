#ifndef MY_WEBCAM_HPP
#define MY_WEBCAM_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <iostream>

class MyWebcam
{
public:
    MyWebcam(const std::string camName, const std::string deviceName, 
        int frameWidth, int frameHeight, int FPS);
    ~MyWebcam();
    int readFrame(cv::Mat& frame, std::string& errMsg);

private:
    cv::VideoCapture cap_;
    std::string camName_;
    std::string deviceName_;
    int frameWidth_;
    int frameHeight_;
    int FPS_;
};

#endif // MY_WEBCAM_HPP
#ifndef MY_HANDS_H
#define MY_HANDS_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct HandResult {
    cv::Rect roi;                        // detection ROI in image coords
    float score = 0.0f;                  // detection confidence
};

class HandTracker {
public:
    // Load YOLO detector (ONNX)
    bool load(const std::string& detectorOnnxPath,
              int detectorInput,
              std::string& err);

    // Backend/target control
    void setBackendTarget(int backend, int target);

    // Run detection on a BGR frame; returns hands
    std::vector<HandResult> infer(const cv::Mat& frameBGR);

private:
    // DNN
    cv::dnn::Net detNet_;

    // Input sizes
    int detSize_ = 640;   // YOLO input (square)

    // State
    int frameCount_ = 0;
    std::vector<cv::Rect> trackedRois_;

    // Smoothing configuration
    float smoothingAlpha_ = 0.3f; // Smoothing factor for EMA
    std::vector<cv::Rect> smoothedRois_; // Smoothed ROIs

    // Pipeline steps
    std::vector<HandResult> runPalmDetector_(const cv::Mat& frameBGR);
};

#endif // MY_HANDS_H
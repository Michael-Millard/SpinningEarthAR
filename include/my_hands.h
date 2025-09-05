#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct HandLandmark {
    cv::Point2f pt; // image coords
    float z = 0.0f; // optional depth-like value from model if available
};

struct HandResult {
    std::vector<HandLandmark> landmarks; // 21 points
    cv::Rect roi;                        // detection ROI in image coords
    float score = 0.0f;                  // detection confidence
    int classId = -1;                    // detector class id (if multi-class)
};

class HandTracker {
public:
    // Load YOLO detector (ONNX) + CMU Caffe hand landmark model
    bool load(const std::string& detectorOnnxPath,
              const std::string& handProtoTxtPath,
              const std::string& handCaffeModelPath,
              int detectorInput,
              int landmarkInput,
              std::string& err);

    // Backend/target control (e.g., DNN_BACKEND_CUDA, DNN_TARGET_CUDA_FP16)
    void setBackendTarget(int backend, int target);

    // Run detection + landmarks on a BGR frame; returns hands (0..2 typical)
    // detectionEvery: run detector every N frames; otherwise reuse last ROIs (no motion model)
    std::vector<HandResult> infer(const cv::Mat& frameBGR, int detectionEvery = 5);

private:
    // DNNs
    cv::dnn::Net detNet_;
    cv::dnn::Net landNet_;

    // Input sizes
    int detSize_ = 640;   // YOLO input (square)
    int lmkSize_ = 368;   // Caffe hand input (square)

    // State
    int frameCount_ = 0;
    std::vector<cv::Rect> trackedRois_;

    // Pipeline steps
    std::vector<cv::Rect> runPalmDetector_(const cv::Mat& frameBGR);
    HandResult runLandmarksOnRoi_(const cv::Mat& frameBGR, const cv::Rect& roi);
};

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

const int POSE_PAIRS[20][2] =
{
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // index
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
};

struct HandLandmark {
    cv::Point2f pt; // image coords
    float z = 0.0f; // optional depth-like value from model if available
};

struct HandResult {
    std::vector<HandLandmark> landmarks; // 21 points
    cv::Rect roi;
    float score = 0.0f;
    bool right = false; // handedness heuristic
};

class HandTracker {
public:
    // Load CMU Caffe hand landmark model (prototxt + caffemodel)
    bool load(const std::string& handProtoTxtPath,
              const std::string & handCaffeModelPath,
              int detectorInput,
              int landmarkInput,
              std::string& err);

    // Backend/target control (e.g., DNN_BACKEND_CUDA, DNN_TARGET_CUDA_FP16)
    void setBackendTarget(int backend, int target);

    // Run detection+landmarks on a BGR frame; returns hands (0..2 typical)
    // detectionEvery: run palm detector every N frames; otherwise track ROIs
    std::vector<HandResult> infer(const cv::Mat& frameBGR, int detectionEvery = 5);

private:
    cv::dnn::Net landNet_;
    int detSize_ = 368;
    int lmkSize_ = 368;
    int frameCount_ = 0;

    std::vector<cv::Rect> trackedRois_;
    std::vector<HandResult> lastHands_;
    
    // Simple centroid-based motion tracking for ROI
    cv::Rect baseROI_;
    bool haveLastCenter_ = false;
    cv::Point2f lastCenter_ = cv::Point2f(0.f, 0.f);
    bool haveLastDelta_ = false;
    cv::Point2f lastDelta_ = cv::Point2f(0.f, 0.f);

    std::vector<cv::Rect> runPalmDetector_(const cv::Mat& frameBGR);
    HandResult runLandmarksOnRoi_(const cv::Mat& frameBGR, const cv::Rect& roi);
};

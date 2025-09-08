#include <my_hands.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace {
// Simple NMS for YOLO style boxes
struct Det { cv::Rect box; float score; int classId; };

static void nms(const std::vector<Det>& in, float iouThresh, std::vector<int>& keep) {
    keep.clear();
    std::vector<int> idx(in.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return in[a].score > in[b].score; });
    std::vector<char> removed(in.size(), 0);
    for (size_t ii = 0; ii < idx.size(); ++ii) {
        int i = idx[ii];
        if (removed[i]) continue;
        keep.push_back(i);
        for (size_t jj = ii + 1; jj < idx.size(); ++jj) {
            int j = idx[jj];
            if (removed[j]) continue;
            float inter = (in[i].box & in[j].box).area();
            float uni = in[i].box.area() + in[j].box.area() - inter;
            float iou = uni > 0 ? inter / uni : 0.f;
            if (iou >= iouThresh) removed[j] = 1;
        }
    }
}
}

bool HandTracker::load(const std::string& detectorOnnxPath,
                       int detectorInput,
                       std::string& err) {
    try {
        // Hand detection network
        detNet_ = cv::dnn::readNet(detectorOnnxPath);
        detNet_.enableFusion(false); // OpenCV was throwing errors in the model fusion function
        if (detNet_.empty()) {
            std::cerr << "Failed to load detector network from: " << detectorOnnxPath << std::endl;
            return false;
        } else {
            std::cout << "Loaded detector network from: " << detectorOnnxPath << std::endl;
        }
        detSize_ = detectorInput > 0 ? detectorInput : 640;
        frameCount_ = 0;
        trackedRois_.clear();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void HandTracker::setBackendTarget(int backend, int target) {
    if (!detNet_.empty()) { 
        detNet_.setPreferableBackend(backend); 
        detNet_.setPreferableTarget(target); 
    }
}

std::vector<HandResult> HandTracker::runPalmDetector_(const cv::Mat& frameBGR) {
    std::vector<HandResult> results;
    if (frameBGR.empty() || detNet_.empty()) return results;

    int inW = detSize_, inH = detSize_;
    // Let DNN handle aspect: letterbox to square
    float r = std::min((float)inW / frameBGR.cols, (float)inH / frameBGR.rows);
    int newW = std::round(frameBGR.cols * r);
    int newH = std::round(frameBGR.rows * r);
    cv::Mat resized; cv::resize(frameBGR, resized, cv::Size(newW, newH));
    cv::Mat input(inH, inW, frameBGR.type(), cv::Scalar(114,114,114)); // YOLO common pad value
    resized.copyTo(input(cv::Rect((inW-newW)/2, (inH-newH)/2, newW, newH)));

    try {
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0/255.0, cv::Size(inW, inH), cv::Scalar(), true, false);
        detNet_.setInput(blob);
        cv::Mat out = detNet_.forward();
        CV_Assert(out.dims == 3 && out.size[0] == 1);

        int ch = out.size[1];      // 5
        int anchors = out.size[2]; // 8400
        CV_Assert(ch == 5);

        cv::Mat chan(ch, anchors, CV_32F, out.ptr<float>());   // (5,8400)
        cv::Mat preds; 
        cv::transpose(chan, preds);                           // (8400,5) rows = candidates

        // Letterbox params from your preprocessing
        int padX = (inW - std::round(frameBGR.cols * r)) / 2;
        int padY = (inH - std::round(frameBGR.rows * r)) / 2;

        const float confThresh = 0.30f;
        for (int i = 0; i < preds.rows; ++i) {
            const float* row = preds.ptr<float>(i); // x,y,w,h,conf
            float conf = row[4];
            if (conf < confThresh) continue;

            float cx = row[0];
            float cy = row[1];
            float w  = row[2];
            float h  = row[3];

            float x1 = cx - 0.5f * w;
            float y1 = cy - 0.5f * h;

            // Remove padding
            x1 -= padX; 
            y1 -= padY;
            // Scale back
            x1 /= r; 
            y1 /= r;
            w  /= r; 
            h  /= r;

            cv::Rect box(
                (int)std::round(x1),
                (int)std::round(y1),
                (int)std::round(w),
                (int)std::round(h)
            );
            box &= cv::Rect(0,0,frameBGR.cols, frameBGR.rows);
            if (box.area() <= 0) continue;

            results.push_back({box, conf});
        }

        // Convert HandResult to Det for NMS
        std::vector<Det> dets;
        for (const auto& result : results) {
            dets.push_back({result.roi, result.score, 0});
        }
        std::vector<int> keep;
        nms(dets, 0.5f, keep);

        // Filter results based on NMS
        std::vector<HandResult> filteredResults;
        for (int k : keep) {
            filteredResults.push_back(results[k]);
        }

        // Update `trackedRois_` to store only ROIs from HandResult
        trackedRois_.clear();
        for (const auto& result : filteredResults) {
            trackedRois_.push_back(result.roi);
        }
        return filteredResults;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
    }
    return results;
}

std::vector<HandResult> HandTracker::infer(const cv::Mat& frameBGR) {
    std::vector<HandResult> hands;
    if (frameBGR.empty()) {
        return hands;
    }

    // Run inference on every frame
    auto handResults = runPalmDetector_(frameBGR);

    // Smooth between consecutive frames
    if (smoothedRois_.size() != handResults.size()) {
        smoothedRois_.resize(handResults.size()); // Resize smoothed ROIs to match current results
    }

    hands.clear();
    for (size_t i = 0; i < handResults.size(); ++i) {
        auto& smoothed = smoothedRois_[i];
        const auto& current = handResults[i].roi;

        // Smooth the center
        smoothed.x = static_cast<int>(smoothingAlpha_ * current.x + (1 - smoothingAlpha_) * smoothed.x);
        smoothed.y = static_cast<int>(smoothingAlpha_ * current.y + (1 - smoothingAlpha_) * smoothed.y);

        // Smooth the size
        smoothed.width = static_cast<int>(smoothingAlpha_ * current.width + (1 - smoothingAlpha_) * smoothed.width);
        smoothed.height = static_cast<int>(smoothingAlpha_ * current.height + (1 - smoothingAlpha_) * smoothed.height);

        hands.push_back({smoothed, handResults[i].score}); // Add smoothed ROI to `hands`
    }

    frameCount_++;
    return hands;
}

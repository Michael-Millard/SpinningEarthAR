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
                       const std::string& handProtoTxtPath,
                       const std::string & handCaffeModelPath,
                       int detectorInput,
                       int landmarkInput,
                       std::string& err) {
    try {
        // Hand detection network
        detNet_ = cv::dnn::readNet(detectorOnnxPath);
        detNet_.enableFusion(false);
        if (detNet_.empty()) {
            std::cerr << "Failed to load detector network from: " << detectorOnnxPath << std::endl;
            return false;
        } else {
            std::cout << "Loaded detector network from: " << detectorOnnxPath << std::endl;
        }
        // Hand landmark network
        landNet_ = cv::dnn::readNetFromCaffe(handProtoTxtPath, handCaffeModelPath);
        landNet_.enableFusion(false);
        if (landNet_.empty()) {
            std::cerr << "Failed to load landmark network from: " << handProtoTxtPath << " and " << handCaffeModelPath << std::endl;
            return false;
        } else {
            std::cout << "Loaded landmark network from: " << handProtoTxtPath << " and " << handCaffeModelPath << std::endl;
        }
        detSize_ = detectorInput > 0 ? detectorInput : 640;
        lmkSize_ = landmarkInput > 0 ? landmarkInput : 368;
        frameCount_ = 0;
        trackedRois_.clear();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void HandTracker::setBackendTarget(int backend, int target) {
    if (!detNet_.empty()) { detNet_.setPreferableBackend(backend); detNet_.setPreferableTarget(target); }
    if (!landNet_.empty()) { landNet_.setPreferableBackend(backend); landNet_.setPreferableTarget(target); }
}

std::vector<cv::Rect> HandTracker::runPalmDetector_(const cv::Mat& frameBGR) {
    std::vector<cv::Rect> rois;
    if (frameBGR.empty() || detNet_.empty()) return rois;

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

        std::vector<Det> dets;
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

            dets.push_back({box, conf, 0}); // class 0 = hand
        }

        // NMS (implement nms to filter by IoU 0.5)
        std::vector<int> keep;
        nms(dets, 0.5f, keep);
        for (int k : keep) {
            rois.push_back(dets[k].box);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
    }
    return rois;
}

HandResult HandTracker::runLandmarksOnRoi_(const cv::Mat& frameBGR, const cv::Rect& roi) {
    HandResult hand; hand.roi = roi;
    if (frameBGR.empty() || landNet_.empty() || roi.area() <= 0) return hand;

    // Clip ROI
    cv::Rect clipped = roi & cv::Rect(0,0, frameBGR.cols, frameBGR.rows);
    if (clipped.area() <= 0) return hand;
    cv::Mat roiImg = frameBGR(clipped).clone();

    // Letterbox to square input
    int inSz = lmkSize_;
    float r = std::min(inSz * 1.f / roiImg.cols, inSz * 1.f / roiImg.rows);
    int newW = std::max(1, (int)std::round(roiImg.cols * r));
    int newH = std::max(1, (int)std::round(roiImg.rows * r));
    cv::Mat resized; cv::resize(roiImg, resized, cv::Size(newW, newH));
    cv::Mat input(inSz, inSz, roiImg.type(), cv::Scalar(0,0,0));
    int padX = (inSz - newW)/2; int padY = (inSz - newH)/2;
    resized.copyTo(input(cv::Rect(padX, padY, newW, newH)));

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0/255.0, cv::Size(inSz, inSz), cv::Scalar(0,0,0), false, false);
    landNet_.setInput(blob);
    cv::Mat out = landNet_.forward();
    if (out.dims != 4) return hand;
    int N = out.size[0];
    int C = out.size[1];
    int H = out.size[2];
    int W = out.size[3];
    if (N != 1 || C < 21) return hand;

    hand.landmarks.reserve(21);
    const float CONF_THRESH = 0.05f;
    for (int k = 0; k < 21; ++k) {
        cv::Mat heat(H, W, CV_32F, out.ptr(0, k));
        double maxVal; cv::Point maxLoc; cv::minMaxLoc(heat, nullptr, &maxVal, nullptr, &maxLoc);
        if (maxVal < CONF_THRESH) { hand.landmarks.push_back({cv::Point2f(-1,-1),0}); continue; }
        float xIn = (maxLoc.x + 0.5f) * (float(inSz)/float(W));
        float yIn = (maxLoc.y + 0.5f) * (float(inSz)/float(H));
        xIn -= padX; yIn -= padY; xIn /= r; yIn /= r;
        float xImg = clipped.x + std::clamp(xIn, 0.f, (float)clipped.width);
        float yImg = clipped.y + std::clamp(yIn, 0.f, (float)clipped.height);
        hand.landmarks.push_back({cv::Point2f(xImg, yImg), 0});
    }
    return hand;
}

std::vector<HandResult> HandTracker::infer(const cv::Mat& frameBGR, int detectionEvery) {
    std::vector<HandResult> hands;
    if (frameBGR.empty()) {
        return hands;
    }
    // Periodic detector; otherwise reuse last trackedRois_
    if (frameCount_ % std::max(1, detectionEvery) == 0 || trackedRois_.empty()) {
        trackedRois_ = runPalmDetector_(frameBGR);
    }
    for (const auto& r : trackedRois_) {
        HandResult h = runLandmarksOnRoi_(frameBGR, r);
        if (!h.landmarks.empty()) {
            hands.push_back(std::move(h));
        }
    }
    if (!trackedRois_.empty()) {
        HandResult h;
        h.roi = trackedRois_[0]; // use the first tracked ROI
        hands.push_back(std::move(h));
    }
    if (smoothingCfg_.enabled && !hands.empty()) {
        applySmoothing_(hands);
    }
    frameCount_++;
    return hands;
}

void HandTracker::applySmoothing_(std::vector<HandResult>& hands) {
    // Match current hands to previous by nearest ROI center
    std::vector<bool> prevUsed(prevStates_.size(), false);
    auto center = [](const cv::Rect& r){ return cv::Point2f(r.x + r.width*0.5f, r.y + r.height*0.5f); };

    std::vector<HandState> newStates; newStates.reserve(hands.size());
    for (auto& h : hands) {
        cv::Point2f c = center(h.roi);
        int bestIdx = -1; float bestDist2 = smoothingCfg_.maxMatchDist * smoothingCfg_.maxMatchDist;
        for (size_t i = 0; i < prevStates_.size(); ++i) {
            if (prevUsed[i]) continue;
            cv::Point2f pc = center(prevStates_[i].roi);
            float dx = c.x - pc.x; float dy = c.y - pc.y; float d2 = dx*dx + dy*dy;
            if (d2 < bestDist2) { bestDist2 = d2; bestIdx = (int)i; }
        }

        HandState state; state.roi = h.roi; state.lm.resize(21);
        if (h.landmarks.size() != 21) { // ensure size
            h.landmarks.resize(21, {cv::Point2f(-1,-1),0});
        }

        if (bestIdx == -1) {
            // New hand: initialize state directly
            for (int i = 0; i < 21; ++i) {
                state.lm[i].pt = h.landmarks[i].pt;
                state.lm[i].stale = (h.landmarks[i].pt.x < 0) ? 1 : 0;
            }
        } else {
            prevUsed[bestIdx] = true;
            auto& prev = prevStates_[bestIdx];
            float a = smoothingCfg_.emaAlpha; // EMA alpha
            for (int i = 0; i < 21; ++i) {
                cv::Point2f cur = h.landmarks[i].pt;
                cv::Point2f prevPt = (i < (int)prev.lm.size()) ? prev.lm[i].pt : cv::Point2f(-1,-1);
                int prevStale = (i < (int)prev.lm.size()) ? prev.lm[i].stale : 0;
                if (cur.x < 0 || cur.y < 0) {
                    // Invalid current: keep previous for limited frames
                    if (prevPt.x >= 0 && prevStale < smoothingCfg_.holdInvalidFrames) {
                        cur = prevPt;
                        prevStale += 1;
                    }
                } else if (prevPt.x >= 0) {
                    // Both valid: smooth
                    cur.x = a * cur.x + (1 - a) * prevPt.x;
                    cur.y = a * cur.y + (1 - a) * prevPt.y;
                    prevStale = 0;
                }
                h.landmarks[i].pt = cur;
                state.lm[i].pt = cur;
                state.lm[i].stale = prevStale;
            }
        }
        newStates.push_back(std::move(state));
    }
    prevStates_ = std::move(newStates);
}

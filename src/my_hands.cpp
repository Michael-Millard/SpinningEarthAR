#include <my_hands.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

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
        detNet_ = cv::dnn::readNet(detectorOnnxPath);
        landNet_ = cv::dnn::readNetFromCaffe(handProtoTxtPath, handCaffeModelPath);
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

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0/255.0, cv::Size(inW, inH), cv::Scalar(), true, false);
    detNet_.setInput(blob);
    cv::Mat out = detNet_.forward();

    // Expecting YOLOv11 ONNX export: [1, N, 85] or [N, 85]
    cv::Mat mat = out;
    if (mat.dims == 3) {
        mat = cv::Mat(out.size[1], out.size[2], CV_32F, out.ptr<float>(0));
    }
    const int numCols = mat.cols; // 85
    const int numRows = mat.rows; // N
    const int xIdx=0,yIdx=1,wIdx=2,hIdx=3, confIdx=4, clsStart=5;

    std::vector<Det> dets;
    const float confThresh = 0.3f; // tune
    for (int i = 0; i < numRows; ++i) {
        const float* row = mat.ptr<float>(i);
        float obj = row[confIdx];
        if (obj < confThresh) continue;
        // If multi-class, pick best class
        int bestId = 0; float bestScore = obj;
        for (int c = clsStart; c < numCols; ++c) {
            float s = row[c] * obj;
            if (s > bestScore) { bestScore = s; bestId = c - clsStart; }
        }
        if (bestScore < confThresh) continue;
        float cx = row[xIdx];
        float cy = row[yIdx];
        float w = row[wIdx];
        float h = row[hIdx];
        // xywh are in input space [0, inW/H]
        float x1 = cx - w*0.5f;
        float y1 = cy - h*0.5f;
        cv::Rect box((int)std::round(x1), (int)std::round(y1), (int)std::round(w), (int)std::round(h));
        // Map from letterboxed input back to original image
        int padX = (inW - newW)/2;
        int padY = (inH - newH)/2;
        box.x -= padX; box.y -= padY;
        box.x = (int)std::round(box.x / r);
        box.y = (int)std::round(box.y / r);
        box.width  = (int)std::round(box.width / r);
        box.height = (int)std::round(box.height/ r);
        box &= cv::Rect(0,0,frameBGR.cols, frameBGR.rows);
        if (box.area() <= 0) continue;
        dets.push_back({box, bestScore, bestId});
    }

    // NMS
    std::vector<int> keep; nms(dets, 0.5f, keep);
    for (int idx : keep) rois.push_back(dets[idx].box);
    return rois;
}

HandResult HandTracker::runLandmarksOnRoi_(const cv::Mat& frameBGR, const cv::Rect& roi) {
    HandResult res; res.roi = roi;
    if (landNet_.empty()) return res;

    // Letterbox ROI to lmkSize_ x lmkSize_
    cv::Mat roiImg = frameBGR(roi);
    int inSz = lmkSize_;
    float r = std::min((float)inSz / roiImg.cols, (float)inSz / roiImg.rows);
    int newW = std::round(roiImg.cols * r);
    int newH = std::round(roiImg.rows * r);
    cv::Mat resized; cv::resize(roiImg, resized, cv::Size(newW, newH));
    cv::Mat input(inSz, inSz, roiImg.type(), cv::Scalar(0,0,0));
    int padX = (inSz - newW)/2; int padY = (inSz - newH)/2;
    resized.copyTo(input(cv::Rect(padX, padY, newW, newH)));

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0/255.0, cv::Size(inSz, inSz), cv::Scalar(0,0,0), false, false);
    landNet_.setInput(blob);
    cv::Mat output = landNet_.forward(); // [1, 22, H, W]

    int H = output.size[2];
    int W = output.size[3];
    const int NUM = output.size[1];
    const float CONF_THRESH = 0.05f;
    res.landmarks.reserve(21);
    for (int n = 0; n < 21; ++n) {
        cv::Mat heat(H, W, CV_32F, output.ptr(0, n));
        double maxVal; cv::Point maxLoc; cv::minMaxLoc(heat, nullptr, &maxVal, nullptr, &maxLoc);
        if (maxVal < CONF_THRESH) { res.landmarks.push_back({cv::Point2f(-1,-1), 0}); continue; }
        // Map heatmap -> input square
        float xIn = ((float)maxLoc.x + 0.5f) * ((float)inSz / (float)W);
        float yIn = ((float)maxLoc.y + 0.5f) * ((float)inSz / (float)H);
        // Undo letterbox
        xIn -= padX; yIn -= padY;
        xIn = xIn / r; yIn = yIn / r;
        // To image coords
        float xImg = roi.x + std::clamp(xIn, 0.f, (float)roi.width);
        float yImg = roi.y + std::clamp(yIn, 0.f, (float)roi.height);
        res.landmarks.push_back({cv::Point2f(xImg, yImg), 0});
    }
    return res;
}

std::vector<HandResult> HandTracker::infer(const cv::Mat& frameBGR, int detectionEvery) {
    std::vector<HandResult> hands; if (frameBGR.empty()) return hands;
    // Periodic detector; otherwise reuse last trackedRois_
    if (frameCount_ % std::max(1, detectionEvery) == 0 || trackedRois_.empty()) {
        trackedRois_ = runPalmDetector_(frameBGR);
    }
    for (const auto& r : trackedRois_) {
        HandResult h = runLandmarksOnRoi_(frameBGR, r);
        if (!h.landmarks.empty()) hands.push_back(std::move(h));
    }
    frameCount_++;
    return hands;
}

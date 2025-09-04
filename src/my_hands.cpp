 #include <my_hands.h>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/dnn.hpp>
 #include <cmath>

bool HandTracker::load(const std::string& handProtoTxtPath,
                       const std::string & handCaffeModelPath,
                       int detectorInput,
                       int landmarkInput,
                       std::string& err) {
    try {
        // CMU/OpenPose hand model (Caffe): single network producing heatmaps for 21 keypoints (+ background)
        landNet_ = cv::dnn::readNetFromCaffe(handProtoTxtPath, handCaffeModelPath);
        detSize_ = detectorInput; // unused for Caffe hand-only; keep for API compatibility
        lmkSize_ = (landmarkInput > 0) ? landmarkInput : 368; // typical input 368x368
        frameCount_ = 0;
        trackedRois_.clear();
        lastHands_.clear();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void HandTracker::setBackendTarget(int backend, int target) {
    if (!landNet_.empty()) {
        landNet_.setPreferableBackend(backend);
        landNet_.setPreferableTarget(target);
    }
}

std::vector<cv::Rect> HandTracker::runPalmDetector_(const cv::Mat& frameBGR) {
    std::vector<cv::Rect> rois;
    if (frameBGR.empty()) return rois;

    // Base ROI anchored at x=50 with height detSize_ and width by aspect ratio
    float aspectRatio = static_cast<float>(frameBGR.cols) / static_cast<float>(frameBGR.rows);
    int detW = static_cast<int>(std::round(aspectRatio * static_cast<float>(detSize_)));
    int detH = detSize_;
    cv::Rect baseRoi(50, (frameBGR.rows - detH) / 2, detW, detH);

    // If we have last frame center, shift ROI by the measured movement
    if (haveLastCenter_) {
        // Predictive delta from last step
        cv::Point2f delta = haveLastDelta_ ? lastDelta_ : cv::Point2f(0.f, 0.f);
        const float alpha = 0.9f; // damping
        const int maxStep = detH / 6; // cap motion per frame
        int dx = static_cast<int>(std::round(alpha * delta.x));
        int dy = static_cast<int>(std::round(alpha * delta.y));
        dx = std::max(-maxStep, std::min(maxStep, dx));
        dy = std::max(-maxStep, std::min(maxStep, dy));
        baseRoi.x += dx;
        baseRoi.y += dy;
    }

    // Clamp ROI within frame
    baseRoi.x = std::max(0, std::min(baseRoi.x, frameBGR.cols - baseRoi.width));
    baseRoi.y = std::max(0, std::min(baseRoi.y, frameBGR.rows - baseRoi.height));

    rois.push_back(baseRoi);
    return rois;
}

HandResult HandTracker::runLandmarksOnRoi_(const cv::Mat& frameBGR, const cv::Rect& roi) {
    HandResult res;
    if (landNet_.empty()) return res;

    // Keep BGR; scale to [0,1]
    cv::Mat netInput = frameBGR(roi).clone();
    int netW = netInput.cols;
    int netH = netInput.rows;
    cv::Mat blob = cv::dnn::blobFromImage(netInput, 1.0/255.0, cv::Size(netW, netH), cv::Scalar(0,0,0), false, false);
    landNet_.setInput(blob);
    cv::Mat output = landNet_.forward();

    int H = output.size[2];
    int W = output.size[3];
    
    // find the position of the body parts
    for (int n = 0; n < 22; n++)
    {
        // Probability map of corresponding body's part.
        cv::Mat probMap(H, W, CV_32F, output.ptr(0,n));
        cv::resize(probMap, probMap, cv::Size(roi.width, roi.height));

        cv::Point maxLoc;
        double prob;
        cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);

        HandLandmark newLandmark;
        newLandmark.pt = (cv::Point2f)maxLoc + cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
        res.landmarks.push_back(newLandmark);
    }

    res.roi = roi;
    return res;
}

std::vector<HandResult> HandTracker::infer(const cv::Mat& frameBGR, int detectionEvery) {
    std::vector<HandResult> hands;
    if (frameBGR.empty()) return hands;

    // Always produce an ROI each frame based on last motion (no periodic reset)
    bool runDetect = trackedRois_.empty();
    if (runDetect) {
        trackedRois_ = runPalmDetector_(frameBGR);
    }

    // Run landmarks on tracked ROIs
    for (const auto& r : trackedRois_) {
        HandResult h = runLandmarksOnRoi_(frameBGR, r);
        if (!h.landmarks.empty()) hands.push_back(std::move(h));
    }

    lastHands_ = hands;
    // Update centroid for next frame ROI shift
    if (!hands.empty() && !hands[0].landmarks.empty()) {
        cv::Point2f sum(0.f, 0.f); int cnt = 0;
        for (const auto& lm : hands[0].landmarks) {
            if (lm.pt.x >= 0 && lm.pt.y >= 0) { sum += lm.pt; cnt++; }
        }
        if (cnt > 0) {
            cv::Point2f center(sum.x / cnt, sum.y / cnt);
            if (!haveLastCenter_) {
                haveLastCenter_ = true;
                lastCenter_ = center;
                haveLastDelta_ = false;
                lastDelta_ = cv::Point2f(0.f, 0.f);
            } else {
                // Update delta and center
                cv::Point2f delta = center - lastCenter_;
                lastDelta_ = delta;
                haveLastDelta_ = true;
                lastCenter_ = center;
            }
        }
    }
    frameCount_++;
    return hands;
}

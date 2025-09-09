#include <my_hands.hpp>

bool HandTracker::load(const std::string& detectorOnnxPath,
                       int detectorInput,
                       bool applySmoothing,
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
        applySmoothing_ = applySmoothing;
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

    // Let DNN handle aspect: letterbox to square
    int inWidth = detSize_, inHeight = detSize_;
    float ratio = std::min((float)inWidth / frameBGR.cols, (float)inHeight / frameBGR.rows);
    int newWidth = std::round(frameBGR.cols * ratio);
    int newHeight = std::round(frameBGR.rows * ratio);
    cv::Mat resized; cv::resize(frameBGR, resized, cv::Size(newWidth, newHeight));
    cv::Mat input(inHeight, inWidth, frameBGR.type(), cv::Scalar(114,114,114)); // YOLO common pad value
    resized.copyTo(input(cv::Rect((inWidth-newWidth)/2, (inHeight-newHeight)/2, newWidth, newHeight)));

    try {
        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0/255.0, cv::Size(inWidth, inHeight), cv::Scalar(), true, false);
        detNet_.setInput(blob);
        cv::Mat output = detNet_.forward();
        CV_Assert(output.dims == 3 && output.size[0] == 1);

        int channels = output.size[1];      // 5
        int anchorCount = output.size[2]; // 8400
        CV_Assert(channels == 5);

        cv::Mat channelData(channels, anchorCount, CV_32F, output.ptr<float>());   // (5,8400)
        cv::Mat predictions; 
        cv::transpose(channelData, predictions);                           // (8400,5) rows = candidates

        // Letterbox params from your preprocessing
        int paddingX = (inWidth - std::round(frameBGR.cols * ratio)) / 2;
        int paddingY = (inHeight - std::round(frameBGR.rows * ratio)) / 2;

        const float nmsThreshold = 0.3f;
        const float confidenceThreshold = 0.8f;
        std::vector<cv::Rect> boundingBoxes;
        std::vector<float> confidenceScores;
        for (int i = 0; i < predictions.rows; ++i) {
            const float* row = predictions.ptr<float>(i); // x,y,w,h,conf
            float confidence = row[4];
            if (confidence < confidenceThreshold) continue;

            float centerX = row[0];
            float centerY = row[1];
            float width  = row[2];
            float height  = row[3];

            float x1 = centerX - 0.5f * width;
            float y1 = centerY - 0.5f * height;

            // Remove padding
            x1 -= paddingX; 
            y1 -= paddingY;
            // Scale back
            x1 /= ratio; 
            y1 /= ratio;
            width /= ratio; 
            height /= ratio;

            cv::Rect boundingBox(
                (int)std::round(x1),
                (int)std::round(y1),
                (int)std::round(width),
                (int)std::round(height)
            );
            boundingBox &= cv::Rect(0,0,frameBGR.cols, frameBGR.rows);
            if (boundingBox.area() <= 0) continue;

            boundingBoxes.push_back(boundingBox);
            confidenceScores.push_back(confidence);
            results.push_back({boundingBox, confidence});
        }

        // NMS
        std::vector<int> keepIndices;
        cv::dnn::NMSBoxes(boundingBoxes, confidenceScores, confidenceThreshold, nmsThreshold, keepIndices);

        // Filter results based on NMS
        std::vector<HandResult> filteredResults;
        for (int k : keepIndices) {
            filteredResults.push_back(results[k]);
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

    if (applySmoothing_) {
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
    } else {
        hands = handResults; // No smoothing; use raw results
    }

    return hands;
}

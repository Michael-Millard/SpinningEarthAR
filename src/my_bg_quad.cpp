#include <my_bg_quad.hpp>
#include <iostream>

BackgroundQuad::BackgroundQuad(const std::string& vertexShaderPath, const std::string& fragmentShaderPath)
    : bgShader_(vertexShaderPath.c_str(), fragmentShaderPath.c_str()), bgVAO_(0), bgVBO_(0), webcamTex_(0), camW_(0), camH_(0) {}

BackgroundQuad::~BackgroundQuad() {
    glDeleteVertexArrays(1, &bgVAO_);
    glDeleteBuffers(1, &bgVBO_);
    glDeleteTextures(1, &webcamTex_);
}

void BackgroundQuad::initialize() {
    // Setup quad vertices
    float vertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 0.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &bgVAO_);
    glGenBuffers(1, &bgVBO_);

    glBindVertexArray(bgVAO_);
    glBindBuffer(GL_ARRAY_BUFFER, bgVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);

    // Setup texture
    glGenTextures(1, &webcamTex_);
    glBindTexture(GL_TEXTURE_2D, webcamTex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void BackgroundQuad::updateTexture(const cv::Mat& frame) {
    if (frame.empty()) return;

    glBindTexture(GL_TEXTURE_2D, webcamTex_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (frame.cols != camW_ || frame.rows != camH_) {
        camW_ = frame.cols;
        camH_ = frame.rows;
        int ch = frame.channels();
        if (ch == 4) { 
            internalFormat_ = GL_RGBA; 
            dataFormat_ = GL_BGRA; 
        } else if (ch == 3) { 
            internalFormat_ = GL_RGB; 
            dataFormat_ = GL_BGR; 
        } else if (ch == 1) { 
            internalFormat_ = GL_RED; 
            dataFormat_ = GL_RED; 
        } else { 
            internalFormat_ = GL_RGB; 
            dataFormat_ = GL_BGR; 
        }
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat_, camW_, camH_, 0, dataFormat_, GL_UNSIGNED_BYTE, frame.data);
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camW_, camH_, dataFormat_, GL_UNSIGNED_BYTE, frame.data);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
}

void BackgroundQuad::render() {
    if (camW_ > 0 && camH_ > 0) {
        glDisable(GL_DEPTH_TEST);
        bgShader_.use();
        bgShader_.setInt("uFrame", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, webcamTex_);
        glBindVertexArray(bgVAO_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glEnable(GL_DEPTH_TEST);
    }
}

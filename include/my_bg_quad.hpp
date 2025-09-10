#ifndef MY_BG_QUAD_HPP
#define MY_BG_QUAD_HPP

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <my_shader.hpp>
#include <opencv2/opencv.hpp>

class BackgroundQuad {
public:
    BackgroundQuad(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
    ~BackgroundQuad();

    void initialize();
    void updateTexture(const cv::Mat& frame);
    void render();

private:
    Shader bgShader_;
    GLuint bgVAO_, bgVBO_, webcamTex_;
    int camW_, camH_;
    GLenum internalFormat_, dataFormat_;
};

#endif

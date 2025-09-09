#ifndef MY_CAMERA_HPP
#define MY_CAMERA_HPP

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

// Constraints on pitch and zoom
const float MIN_PITCH = -89.0f;
const float MAX_PITCH = 89.0f;
const float MIN_ZOOM = 1.0f;
const float MAX_ZOOM = 60.0f;

// Default camera values
const float CAMERA_SPEED = 2.5f;
const float MOUSE_SENSITIVITY = 0.1f;
const float ZOOM = 50.0f; // FOV

class Camera
{
public:
    // Camera attributes
    glm::vec3 position_;
    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;
    glm::vec3 worldUp_;

    // Euler Angles
    float yaw_;
    float pitch_;

    // Camera params
    float movementSpeed_;
    float mouseSensitivity_;
    float zoom_;
    bool fixedHeight_;
    float fixedYPos_;
    bool zoomEnabled_;

    // Constructor
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f),
        float yaw = -90.0f, float pitch = 0.0f, bool fixedHeight = false, float yFixed = 0.0f, bool zoomEnabled = true)
        : position_(position), worldUp_(worldUp), yaw_(yaw), pitch_(pitch), fixedHeight_(fixedHeight), fixedYPos_(yFixed), zoomEnabled_(zoomEnabled) {
        // Set front, right, up vectors
        front_ = glm::vec3(0.0f, 0.0f, -1.0f);
        right_ = glm::normalize(glm::cross(front_, worldUp_));
        up_ = glm::normalize(glm::cross(right_, front_));

        // Remaining params
        movementSpeed_ = CAMERA_SPEED;
        mouseSensitivity_ = MOUSE_SENSITIVITY;
        zoom_ = ZOOM;
    }

    // Set initial position
    void setPosition(const glm::vec3& newPos) {
        position_ = newPos;
    }

    // Set sensitivity
    void setMouseSensitivity(const float newSensitivity) {
        mouseSensitivity_ = newSensitivity;
    }

    // Set camera movement speed
    void setCameraMovementSpeed(const float newSpeed) {
        movementSpeed_ = newSpeed;
    }

    // Set fixed height camera
    void setFixedHeightCamera(const bool fixedHeight, const float yPos) {
        fixedHeight_ = fixedHeight;
        fixedYPos_ = yPos;
    }

    // Set zoom
    void setZoom(const float zoom) {
        zoom_ = zoom;
    }

    // Enable/Disable zoom
    void setZoomEnabled(const bool enable) {
        zoomEnabled_ = enable;
    }

    // Returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 getViewMatrix() {
        return glm::lookAt(position_, position_ + front_, up_);
    }

    // Processes input received from keyboard
    void processKeyboardInput(const int direction, float deltaTime) {
        float velocity = movementSpeed_ * deltaTime;
        if (direction == GLFW_KEY_W) {
            position_ += front_ * velocity;
        } else if (direction == GLFW_KEY_A) {
            position_ -= right_ * velocity;
        } else if (direction == GLFW_KEY_S) {
            position_ -= front_ * velocity;
        } else if (direction == GLFW_KEY_D) {
            position_ += right_ * velocity;
        } else if (direction == GLFW_KEY_Q) {
            position_ += up_ * velocity;
        } else if (direction == GLFW_KEY_E) {
            position_ -= up_ * velocity;
        }

        // If fixed height camera, ignore y-coordinate changes
        if (fixedHeight_) {
            position_.y = fixedYPos_;
        }
    }


    // Processes input received from mouse
    void processMouseMovement(float xOff, float yOff) {
        xOff *= mouseSensitivity_;
        yOff *= mouseSensitivity_;
        yaw_ += xOff;
        pitch_ += yOff;

        // Constrain pitch
        if (pitch_ > MAX_PITCH) {
            pitch_ = MAX_PITCH;
        }
        if (pitch_ < MIN_PITCH) {
            pitch_ = MIN_PITCH;
        }

        // Update front, right and up vectors with updated Euler angles
        updateCameraVectors();
    }

    // Processes input received from mouse scroll-wheel
    void processMouseScroll(float yOff) {
        // Only if zoom is enabled
        if (zoomEnabled_) {
            zoom_ -= yOff;
            // Constrain zoom
            if (zoom_ < MIN_ZOOM) {
                zoom_ = MIN_ZOOM;
            }
            if (zoom_ > MAX_ZOOM) {
                zoom_ = MAX_ZOOM;
            }
        }
    }

private:
    // Calculates the front vector from camera's new Euler Angles
    void updateCameraVectors()
    {
        // Front vector
        glm::vec3 newFront;
        newFront.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        newFront.y = sin(glm::radians(pitch_));
        newFront.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front_ = glm::normalize(newFront);

        // Right and up vectors
        right_ = glm::normalize(glm::cross(front_, worldUp_));
        up_ = glm::normalize(glm::cross(right_, front_));
    }
};
#endif // MY_CAMERA_H


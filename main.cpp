#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <my_shader.hpp>
#include <my_model.hpp>
#include <my_camera.hpp>
#include <my_webcam.hpp>
#include <my_hands.hpp>
#include <my_cli.hpp>
#include <my_bg_quad.hpp>

#include <iostream>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Callback function declarations
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
void processUserInput(GLFWwindow* window);

// Global params for callback functions
int screenWidth;
int screenHeight;
float earthScale;

int setupGLFW(GLFWwindow** window) {
    // glfw init and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE); 

    // glfw window creation
    GLFWwindow* glfw_window = glfwCreateWindow(
        screenWidth,
        screenHeight,
        "Globe",
        nullptr,  // windowed (no monitor for fullscreen)
        nullptr
    );
    if (glfw_window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(glfw_window);

    // Callback functions
    glfwSetFramebufferSizeCallback(glfw_window, frameBufferSizeCallback);

    // Show cursor, for debugging
    glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Load all OpenGL function pointers with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    *window = glfw_window;

    // Configure global OpenGL state
    glEnable(GL_DEPTH_TEST);    // Depth-testing
    glDepthFunc(GL_LESS);       // Smaller value as "closer" for depth-testing
    glEnable(GL_CULL_FACE);     // Cull back faces to reduce fragment work
    glCullFace(GL_BACK);    
    glFrontFace(GL_CCW);

    // Initialize viewport to current framebuffer size
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(glfw_window, &fbw, &fbh);
    if (fbw > 0 && fbh > 0) {
        glViewport(0, 0, fbw, fbh);
        screenWidth = static_cast<unsigned int>(fbw);
        screenHeight = static_cast<unsigned int>(fbh);
    }
    return 0;
}

// Inputs: view, proj (glm::mat4), winW, winH, palmWinPx (window pixels, origin top-left), planeDist d
glm::vec3 screenToWorldOnPlane(glm::mat4 view, glm::mat4 proj,
                               int winW, int winH,
                               glm::vec2 palmWinPx, float planeDist) {
    glm::ivec4 viewport(0, 0, winW, winH);

    // glm::unProject expects origin at bottom-left: flip Y
    glm::vec3 winNear(palmWinPx.x, float(winH) - palmWinPx.y, 0.0f);
    glm::vec3 winFar (palmWinPx.x, float(winH) - palmWinPx.y, 1.0f);

    glm::vec3 pNear = glm::unProject(winNear, view, proj, viewport);
    glm::vec3 pFar  = glm::unProject(winFar,  view, proj, viewport);
    glm::vec3 dir   = glm::normalize(pFar - pNear);

    // Camera world pose and forward
    glm::mat4 invV = glm::inverse(view);
    glm::vec3 camPos = glm::vec3(invV[3]);
    glm::vec3 camFwd = glm::normalize(glm::vec3(invV * glm::vec4(0, 0, -1, 0))); // -Z in view space

    // Plane: point at camPos + d*camFwd, normal = camFwd
    glm::vec3 planePoint = camPos + camFwd * planeDist;
    float denom = glm::dot(dir, camFwd);
    if (fabs(denom) < 1e-6f) return pNear; // nearly parallel; fallback

    float t = glm::dot(planePoint - pNear, camFwd) / denom;
    return pNear + t * dir;
}

int main(int argc, char** argv) {
    // Parse CLI arguments
    CLIOptions options = parseCli(argc, argv);
    if (options.show_help) {
        printHelp(argv[0]);
        return 0;
    }

    // Set global params
    screenWidth = options.screenWidth;
    screenHeight = options.screenHeight;
    earthScale = options.earthScale;

    // Window
    GLFWwindow* window = nullptr;
    if (setupGLFW(&window) < 0) {
        std::cerr << "Failed to setup GLFW. Exiting.\n";
        return -1;
    }

    // Shaders
    Shader earthShader("shaders/earth_shader.vs", "shaders/earth_shader.fs");
    Shader bgShader("shaders/bg_quad.vs", "shaders/bg_quad.fs");

    // Models
    Model earthModel(options.earthModelPath, "Earth");
    Model moonModel(options.moonModelPath, "Moon");
    Model spitfireModel(options.spitfireModelPath, "Spitfire");

    // Virtual camera
    Camera camera;
    camera.setPosition(options.initPosition);
    camera.setMouseSensitivity(options.mouseSensitivity);
    camera.setCameraMovementSpeed(options.cameraSpeed);
    camera.setZoom(options.cameraZoom);
    camera.setFixedHeightCamera(false, options.initPosition.y);
    camera.setZoomEnabled(false);

    // Webcam (For device name, run: $ v4l2-ctl --list-devices)
    MyWebcam webcam(options.webcamName, options.deviceName, screenWidth, screenHeight, options.fps);
    cv::Mat currentFrame(cv::Size(screenWidth, screenHeight), CV_8UC3);
    std::string errMsg;
    int initRead = webcam.readFrame(currentFrame, errMsg);
    if (initRead != 0) {
        std::cerr << "Warning: " << errMsg << " (continuing; will retry each frame)" << std::endl;
        currentFrame.release();
    }

    // Hand tracker setup 
    HandTracker handTracker;
    std::string handErr;
    bool handsReady = handTracker.load(options.onnxModelPath, screenWidth, options.applySmoothing, handErr);
    if (!handsReady) {
        std::cerr << "HandTracker load failed: " << handErr << std::endl;
        return -1;
    } else {
        // Prefer GPU if available; else fall back to default
        handTracker.setBackendTarget(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);
    }

    // Background Quad
    BackgroundQuad bgQuad("shaders/bg_quad.vs", "shaders/bg_quad.fs");
    bgQuad.initialize();

    // Render loop
    float yRot = 0.0f;
    float deltaTime = 0.0f;
    float prevFrame = 0.0f;
    float elapsedTime = 0.0f;
    glm::vec3 lastEarthPos = glm::vec3(0.0f);
    cv::Point2i prevPalmPos(screenWidth / 2, screenHeight / 2);
    while (!glfwWindowShouldClose(window))
    {
        // Clear screen colour and buffers
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Per-frame time logic
        float currentTime = static_cast<float>(glfwGetTime());
        deltaTime = currentTime - prevFrame;
        elapsedTime += deltaTime;
        prevFrame = currentTime;

        // Rotate the model slowly about the y-axis
        yRot += 20.0f * deltaTime;
        yRot = fmodf(yRot, 360.0f);

        // Exit on ESC
        processUserInput(window);

        // Update webcam texture (and optionally overlay hands) at most ~30 fps
        std::vector<HandResult> hands;
        if (webcam.readFrame(currentFrame, errMsg) == 0) {
            // Run hand tracker on the fresh frame
            hands = handTracker.infer(currentFrame);
        }

        // Update webcam texture
        bgQuad.updateTexture(currentFrame);

        // Render background quad
        bgQuad.render();

        // Setup uniforms in shaders
        glm::mat4 model = glm::identity<glm::mat4>();
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.zoom_),
            static_cast<float>(screenWidth) / static_cast<float>(screenHeight), 
            0.1f, 1000.0f);

        // Updated logic to use the center of the detected hand ROI
        if (!hands.empty()) {
            // Only do one hand for now: highest confidence score
            HandResult bestHand = hands[0];
            for (const auto& hr : hands) {
                if (hr.score > bestHand.score) { 
                    bestHand = hr;
                }
            }
            
            // Check if plausible match (nearby to previous palm, starting in centre of frame)
            cv::Point2i handPalmPos = bestHand.roi.tl() + cv::Point2i(bestHand.roi.width / 2, bestHand.roi.height / 2);
            float distToPrevPalm = cv::norm(handPalmPos - prevPalmPos);
            if (distToPrevPalm < 100.0f) {
                // Use bounding box center of best hand as palm point
                if (handPalmPos.x >= 0 && handPalmPos.y >= 0 && handPalmPos.x < currentFrame.cols && handPalmPos.y < currentFrame.rows) {
                    glm::vec2 palmVideoPx(handPalmPos.x, handPalmPos.y - 15); // Small nudge higher (+Y axis is down in image coords)
                    glm::ivec2 winSize(screenWidth, screenHeight);
                    glm::vec2 palmWinPx = palmVideoPx; // assuming webcam fills window; adjust if letterboxed
                    glm::vec3 worldPos = screenToWorldOnPlane(view, projection, winSize.x, winSize.y, palmWinPx, options.initPosition.z);
                    lastEarthPos = worldPos;
                }
                prevPalmPos = handPalmPos;
            }
        } 
        glm::vec3 worldPos = lastEarthPos;

        // Slightly scale down to keep fully within the frame
        model = glm::scale(model, glm::vec3(earthScale));
        model = glm::rotate(model, glm::radians(yRot), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::translate(glm::mat4(1.0f), worldPos) * model;

        // Set shader uniforms
        earthShader.use();
        earthShader.setMat4("model", model);
        earthShader.setMat4("view", view);
        earthShader.setMat4("projection", projection);

        // Lighting uniforms for current shader (world space)
        earthShader.setVec3("lightPos", glm::vec3(5.0f, 0.0f, 5.0f)); 
        earthShader.setVec3("viewPos", camera.position_);
        earthShader.setFloat("shininess", 32.0f);
        earthModel.draw(earthShader);

        // Earth transform without scale: translation to worldPos and Earth rotation
        glm::mat4 earthTR = glm::translate(glm::mat4(1.0f), worldPos) *
                            glm::rotate(glm::mat4(1.0f), glm::radians(yRot), glm::vec3(0.0f, 1.0f, 0.0f));

        // Orbit in Earth's local XZ plane (equator)
        float theta = glm::radians(elapsedTime * options.spitfireOrbitSpeedDeg);
        glm::vec3 orbitPos = glm::vec3(
            options.spitfireOrbitRadius * cosf(theta),
            0.0f,
            options.spitfireOrbitRadius * sinf(theta)
        );

        // Tangent direction along the orbit (forward direction) in Earth-local frame
        glm::vec3 forward = glm::normalize(glm::vec3(-sinf(theta), 0.0f, cosf(theta)));
        glm::vec3 up(0.0f, 1.0f, 0.0f); // Earth's up
        glm::vec3 right = glm::normalize(glm::cross(forward, up));

        // Recompute up to ensure orthonormal basis
        up = glm::normalize(glm::cross(right, forward));

        // Columns are the basis vectors (Earth-local): right, up, forward
        glm::mat4 basis(1.0f);
        basis[0] = glm::vec4(right, 0.0f);
        basis[1] = glm::vec4(up, 0.0f);
        basis[2] = glm::vec4(forward, 0.0f);

        // Compose spitfire relative to Earth: Earth TR -> orbit translate -> orientation -> local roll -> scale
        glm::mat4 planeModel = earthTR
            * glm::translate(glm::mat4(1.0f), orbitPos)
            * basis
            * glm::rotate(glm::mat4(1.0f), glm::radians(-45.0f), glm::vec3(0.0f, 0.0f, 1.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(options.spitfireScale));
        earthShader.setMat4("model", planeModel);

        // Apply per-mesh transform to spin propeller meshes
        float propAngle = (2.0f * M_PI) * options.propellerRps * elapsedTime; // radians
        spitfireModel.drawWithTransforms(earthShader, [&](const std::string& meshName) -> glm::mat4 {
            std::string lower = meshName;
            for (char& c : lower) c = static_cast<char>(::tolower(c));
            if (lower.find("prop") != std::string::npos) {
                return glm::rotate(glm::mat4(1.0f), propAngle, options.propellerAxis);
            }
            return glm::mat4(1.0f);
        }); 

        // Render four Spitfires spaced 90 degrees apart
        for (int i = 0; i < 4; ++i) {
            float angleOffset = glm::radians(90.0f * i);
            float adjustedTheta = theta + angleOffset;

            glm::vec3 adjustedOrbitPos = glm::vec3(
                options.spitfireOrbitRadius * cosf(adjustedTheta),
                0.0f,
                options.spitfireOrbitRadius * sinf(adjustedTheta)
            );

            glm::vec3 adjustedForward = glm::normalize(glm::vec3(-sinf(adjustedTheta), 0.0f, cosf(adjustedTheta)));
            glm::vec3 adjustedRight = glm::normalize(glm::cross(adjustedForward, up));
            glm::vec3 adjustedUp = glm::normalize(glm::cross(adjustedRight, adjustedForward));

            glm::mat4 adjustedBasis(1.0f);
            adjustedBasis[0] = glm::vec4(adjustedRight, 0.0f);
            adjustedBasis[1] = glm::vec4(adjustedUp, 0.0f);
            adjustedBasis[2] = glm::vec4(adjustedForward, 0.0f);

            glm::mat4 adjustedPlaneModel = earthTR
                * glm::translate(glm::mat4(1.0f), adjustedOrbitPos)
                * adjustedBasis
                * glm::rotate(glm::mat4(1.0f), glm::radians(-45.0f), glm::vec3(0.0f, 0.0f, 1.0f))
                * glm::scale(glm::mat4(1.0f), glm::vec3(options.spitfireScale));

            earthShader.setMat4("model", adjustedPlaneModel);

            spitfireModel.drawWithTransforms(earthShader, [&](const std::string& meshName) -> glm::mat4 {
                std::string lower = meshName;
                for (char& c : lower) c = static_cast<char>(::tolower(c));
                if (lower.find("prop") != std::string::npos) {
                    return glm::rotate(glm::mat4(1.0f), propAngle, options.propellerAxis);
                }
                return glm::mat4(1.0f);
            });
        }

        // Moon orbit parameters
        float moonTheta = glm::radians(elapsedTime * options.moonOrbitSpeedDeg);
        glm::vec3 moonOrbitPos = glm::vec3(
            options.moonOrbitRadius * cosf(moonTheta),
            0.0f,
            options.moonOrbitRadius * sinf(moonTheta)
        );

        // Compute moon orientation to always face Earth
        glm::vec3 moonForward = glm::normalize(-moonOrbitPos); // Point towards Earth
        glm::vec3 moonRight = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), moonForward));
        glm::vec3 moonUp = glm::normalize(glm::cross(moonForward, moonRight));

        glm::mat4 moonBasis(1.0f);
        moonBasis[0] = glm::vec4(moonRight, 0.0f);
        moonBasis[1] = glm::vec4(moonUp, 0.0f);
        moonBasis[2] = glm::vec4(moonForward, 0.0f);

        glm::mat4 moonModelMatrix = earthTR
            * glm::translate(glm::mat4(1.0f), moonOrbitPos)
            * moonBasis
            * glm::scale(glm::mat4(1.0f), glm::vec3(options.moonScale));

        // Ensure the moon model uses its own texture
        earthShader.setMat4("model", moonModelMatrix);
        moonModel.draw(earthShader);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up and exit
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// Process keyboard inputs
void processUserInput(GLFWwindow* window) {
    // Escape to exit
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    } 
}

// Window size change callback
void frameBufferSizeCallback(GLFWwindow* window, int width, int height) {
    // Prevent zero dimension viewport
    if (width == 0 || height == 0) {
        return;
    }
    // Ensure viewport matches new window dimensions
    glViewport(0, 0, width, height);

    // Adjust screen width and height params that set the aspect ratio in the projection matrix
    screenWidth = width;
    screenHeight = height;
}
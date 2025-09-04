#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <my_shader.h>
#include <my_model.h>
#include <my_camera.h>
#include <my_webcam.h>
#include <my_hands.h>

#include <iostream>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// For cv::cvtColor
#include <opencv2/imgproc.hpp>

// Callback function declarations
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
void processUserInput(GLFWwindow* window);

// Overlay: draw detected hands on a BGR frame
static void drawHandsOverlay(cv::Mat& frame, const std::vector<HandResult>& hands) {
    if (frame.empty()) return;
    // Edge list for 21-point OpenPose hand model
    static const int EDGES[][2] = {
        {0,1},{1,2},{2,3},{3,4},      // thumb
        {0,5},{5,6},{6,7},{7,8},      // index
        {0,9},{9,10},{10,11},{11,12}, // middle
        {0,13},{13,14},{14,15},{15,16}, // ring
        {0,17},{17,18},{18,19},{19,20}  // little
    };
    const int EDGE_COUNT = sizeof(EDGES)/sizeof(EDGES[0]);
    const cv::Scalar kCircleColor(208, 224, 64); // turquoise (B,G,R)
    const cv::Scalar kLineColor(20, 255, 57);    // lime green (B,G,R)

    auto validPt = [&](const cv::Point2f& p){ return p.x >= 0 && p.y >= 0 && p.x < frame.cols && p.y < frame.rows; };
    for (const auto& h : hands) {
        if (h.landmarks.size() >= 21) {
            // Draw lines only if both endpoints are valid
            for (int i = 0; i < EDGE_COUNT; ++i) {
                const auto& a = h.landmarks[EDGES[i][0]].pt;
                const auto& b = h.landmarks[EDGES[i][1]].pt;
                if (validPt(a) && validPt(b)) {
                    cv::line(frame, a, b, kLineColor, 2, cv::LINE_AA);
                }
            }
            // Draw joints only if valid
            for (int i = 0; i < 21; ++i) {
                const auto& p = h.landmarks[i].pt;
                if (validPt(p)) {
                    cv::circle(frame, p, 4, kCircleColor, -1, cv::LINE_AA);
                }
            }
        }
    }
}

// Screen params
unsigned int SCREEN_WIDTH = 640;
unsigned int SCREEN_HEIGHT = 480;

// Timing params
float delta_time = 0.0f;
float prev_frame = 0.0f;
float elapsed_time = 0.0f;

// Hand tracking toggle
bool HANDS_ENABLED = true;

// Model names
#define EARTH_MODEL "models/earth.obj"
#define SPITFIRE_MODEL "models/spitfire.obj"

// Spitfire orbit params
const float PLANE_ORBIT_RADIUS = 3.5f;     // distance from Earth's center
const float PLANE_ORBIT_SPEED_DEG = 30.0f; // degrees per second
const float PLANE_SCALE = 0.25f;           // relative size vs Earth

// Propeller animation params
const float PROPELLER_RPS = 2.0f;         // revolutions per second
const glm::vec3 PROPELLER_AXIS = glm::vec3(0.0f, 0.21443f, 3.382f); // local Z axis

// Model matrix params
float y_rot = 0.0f;

// Camera specs (set later, can't call functions here)
const float camera_speed = 3.0f;
const float mouse_sensitivity = 0.1f;
const float camera_zoom = 50.0f;
const float x_pos_init = 0.0f;
const float y_pos_init = 0.0f;
const float z_pos_init = 15.0f;
Camera camera(glm::vec3(x_pos_init, y_pos_init, z_pos_init));

int setupGLFW(GLFWwindow** window) {
    // glfw init and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE); // Show title bar (windowed)

    // Screen params
    //GLFWmonitor* my_monitor = glfwGetPrimaryMonitor();
    //const GLFWvidmode* mode = glfwGetVideoMode(my_monitor);
    //SCREEN_WIDTH = mode->width; 
    //SCREEN_HEIGHT = mode->height;

    // glfw window creation
    GLFWwindow* glfw_window = glfwCreateWindow(
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
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
    // Disable vsync to maximize FPS (you can set to 1 to re-enable)
    glfwSwapInterval(1);

    // Callback functions
    glfwSetFramebufferSizeCallback(glfw_window, frameBufferSizeCallback);

    // Disable cursor
    glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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
        SCREEN_WIDTH = static_cast<unsigned int>(fbw);
        SCREEN_HEIGHT = static_cast<unsigned int>(fbh);
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

int main() {
    // Window
    GLFWwindow* window = nullptr;
    if (setupGLFW(&window)) {
        std::cerr << "Failed to setup GLFW. Exiting.\n";
        return -1;
    }

    // Shaders
    Shader earth_shader("shaders/earth_shader.vs", "shaders/earth_shader.fs");
    Shader bg_shader("shaders/bg_quad.vs", "shaders/bg_quad.fs");

    // Models
    Model earth_model(EARTH_MODEL, "Earth");
    Model spitfire_model(SPITFIRE_MODEL, "Spitfire");

    // Camera
    camera.setMouseSensitivity(mouse_sensitivity);
    camera.setCameraMovementSpeed(camera_speed);
    camera.setZoom(camera_zoom);
    camera.setFPSCamera(false, y_pos_init);
    camera.setZoomEnabled(false);

    // Webcam (For device name, run: $ v4l2-ctl --list-devices)
    MyWebcam webcam("Webcam", "/dev/video2", SCREEN_WIDTH, SCREEN_HEIGHT);
    cv::Mat current_frame;
    std::string err_msg;
    int initRead = webcam.ReadFrame(current_frame, err_msg);
    if (initRead != 0) {
        std::cerr << "Warning: " << err_msg << " (continuing; will retry each frame)" << std::endl;
        current_frame.release();
    }

    // --- Setup fullscreen background quad and webcam texture ---
    GLuint bgVAO = 0, bgVBO = 0, webcamTex = 0;
    // Fullscreen quad (NDC) using triangle strip; flip V in texcoords to match OpenCV's top-left origin
    float bgQuad[] = {
        // x, y,   u, v
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 0.0f,
    };
    glGenVertexArrays(1, &bgVAO);
    glGenBuffers(1, &bgVBO);
    glBindVertexArray(bgVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bgVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bgQuad), bgQuad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Create the webcam texture; allocate lazily when first valid frame arrives
    glGenTextures(1, &webcamTex);
    glBindTexture(GL_TEXTURE_2D, webcamTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    int camW = 0;
    int camH = 0;
    GLenum internalFormat = GL_RGB;
    GLenum dataFormat = GL_RGB;
    if (initRead == 0) {
        glBindTexture(GL_TEXTURE_2D, webcamTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        camW = current_frame.cols;
        camH = current_frame.rows;
        int ch = current_frame.channels();
        if (ch == 4) { internalFormat = GL_RGBA; dataFormat = GL_BGRA; }
        else if (ch == 3) { internalFormat = GL_RGB; dataFormat = GL_BGR; }
        else if (ch == 1) { internalFormat = GL_RED; dataFormat = GL_RED; }
        else { internalFormat = GL_RGB; dataFormat = GL_BGR; }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, camW, camH, 0, dataFormat, GL_UNSIGNED_BYTE, current_frame.data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Throttle webcam updates to this interval (seconds)
    const float CAM_UPDATE_INTERVAL = 1.0f / 30.0f; // ~30 FPS
    float lastCamUpdateTime = 0.0f;

    // Hand tracker setup (models need to be provided by you; ONNX paths)
    HandTracker handTracker;
    std::string handErr;
    // TODO: Replace with actual ONNX paths (palm detector + landmark models)
    std::string protoTxt = "caffe/hands/pose_deploy.prototxt";
    std::string caffeModel = "caffe/hands/pose_iter_102000.caffemodel";
    bool handsReady = handTracker.load(protoTxt, caffeModel, 256, 224, handErr);
    if (!handsReady) {
        std::cerr << "HandTracker load failed: " << handErr << std::endl;
        HANDS_ENABLED = false;
    } else {
        // Prefer GPU if available; else fall back to default
        handTracker.setBackendTarget(cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);
    }

    // Render loop
    while (!glfwWindowShouldClose(window))
    {
        // Clear screen colour and buffers
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Per-frame time logic
        float current_time = static_cast<float>(glfwGetTime());
        delta_time = current_time - prev_frame;
        elapsed_time += delta_time;
        prev_frame = current_time;

        // Rotate the model slowly about the y-axis
        y_rot += 20.0f * delta_time;
        y_rot = fmodf(y_rot, 360.0f);

        // Exit on ESC
        processUserInput(window);

        // Update webcam texture (and optionally overlay hands) at most ~30 fps
        std::vector<HandResult> hands;
        if ((elapsed_time - lastCamUpdateTime) >= CAM_UPDATE_INTERVAL) {
            if (webcam.ReadFrame(current_frame, err_msg) == 0) {
                // Run hand tracker on the fresh frame
                if (HANDS_ENABLED) {
                    hands = handTracker.infer(current_frame, 5);
                    // Draw overlay directly on the frame so it appears in the background texture
                    drawHandsOverlay(current_frame, hands);
                }

                // Upload to GL texture
                glBindTexture(GL_TEXTURE_2D, webcamTex);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                if (current_frame.cols != camW || current_frame.rows != camH) {
                    camW = current_frame.cols;
                    camH = current_frame.rows;
                    int ch = current_frame.channels();
                    if (ch == 4) { internalFormat = GL_RGBA; dataFormat = GL_BGRA; }
                    else if (ch == 3) { internalFormat = GL_RGB; dataFormat = GL_BGR; }
                    else if (ch == 1) { internalFormat = GL_RED; dataFormat = GL_RED; }
                    else { internalFormat = GL_RGB; dataFormat = GL_BGR; }
                    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, camW, camH, 0, dataFormat, GL_UNSIGNED_BYTE, current_frame.data);
                } else {
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camW, camH, dataFormat, GL_UNSIGNED_BYTE, current_frame.data);
                }
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            lastCamUpdateTime = elapsed_time;
        }

        // Draw background quad only if we have a valid frame size
        if (camW > 0 && camH > 0) {
            glDisable(GL_DEPTH_TEST);
            bg_shader.use();
            bg_shader.setInt("uFrame", 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, webcamTex);
            glBindVertexArray(bgVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);

            // Optional: overlay simple landmark points using gl_Point primitives
            // For now, skip custom GL overlay; could convert to a small dynamic VBO later.
            glEnable(GL_DEPTH_TEST);
        }

        // View and projection
        //camera.position = glm::vec3(0.0f, 0.0f, 8.0f);
        
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.zoom),
            static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT), 
            0.1f, 1000.0f);

        // All have same model, view and projection 
        glm::mat4 model = glm::identity<glm::mat4>();

        // Input: pixel (x, y) from hand landmark
        float x = hands[0].landmarks[9].pt.x;
        float y = hands[0].landmarks[9].pt.y;

        // TRANSLATE EARTH TO CENTRE OF HAND USING (x, y)
        // Get the 9th landmark (be careful: 0-based index is 8 if your list is 0-based)
        glm::vec2 palmVideoPx = glm::vec2(x, y);

        // Map to window pixels (or skip if not letterboxed)
        glm::ivec2 videoSize(camW, camH);     // your webcam frame size
        glm::ivec2 winSize(SCREEN_WIDTH, SCREEN_HEIGHT); // your GLFW window size
        //glm::vec2 palmWinPx = videoPxToWindowPx(videoSize, winSize, palmVideoPx);
        glm::vec2 palmWinPx = palmVideoPx; // Assume no letterboxing for simplicity
        // Choose a plane distance in front of camera (e.g., where you’d like the globe to live)
        glm::vec3 worldPos = screenToWorldOnPlane(view, projection, winSize.x, winSize.y, palmWinPx, z_pos_init);


        // Slightly scale down to keep fully within the frame
        model = glm::scale(model, glm::vec3(0.8f));
        model = glm::rotate(model, glm::radians(y_rot), glm::vec3(0.0f, 1.0f, 0.0f));

        // Set globe transform
        model = glm::translate(glm::mat4(1.0f), worldPos) * model; // keep your existing rotation/scale

        earth_shader.use();
        earth_shader.setMat4("model", model);
        earth_shader.setMat4("view", view);
        earth_shader.setMat4("projection", projection);

        // Subtle indoor-light uniforms
        // Lighting uniforms for current shader (world space)
        earth_shader.setVec3("lightPos", glm::vec3(3.0f, 0.0f, 5.0f)); // from left, above, slightly forward
        earth_shader.setVec3("viewPos", camera.position);
        earth_shader.setFloat("shininess", 32.0f);

        // Draw the Earth
        earth_model.draw(earth_shader);

        // --- Draw the Spitfire orbiting the Earth's equator (relative to Earth) ---
        // Earth transform without scale: translation to worldPos and Earth rotation
        glm::mat4 earthTR = glm::translate(glm::mat4(1.0f), worldPos) *
                            glm::rotate(glm::mat4(1.0f), glm::radians(y_rot), glm::vec3(0.0f, 1.0f, 0.0f));

        // Orbit in Earth's local XZ plane (equator)
        float theta = glm::radians(elapsed_time * PLANE_ORBIT_SPEED_DEG);
        glm::vec3 orbitPos = glm::vec3(
            PLANE_ORBIT_RADIUS * cosf(theta),
            0.0f,
            PLANE_ORBIT_RADIUS * sinf(theta)
        );

        // Tangent direction along the orbit (forward direction) in Earth-local frame
        glm::vec3 forward = glm::normalize(glm::vec3(
            -sinf(theta), 0.0f, cosf(theta)
        ));
        glm::vec3 up(0.0f, 1.0f, 0.0f); // Earth's up
        glm::vec3 right = glm::normalize(glm::cross(forward, up));
        // Recompute up to ensure orthonormal basis
        up = glm::normalize(glm::cross(right, forward));

        glm::mat4 basis(1.0f);
        // Columns are the basis vectors (Earth-local): right, up, forward
        basis[0] = glm::vec4(right, 0.0f);
        basis[1] = glm::vec4(up, 0.0f);
        basis[2] = glm::vec4(forward, 0.0f);

        // Compose plane relative to Earth: Earth TR -> orbit translate -> orientation -> local roll -> scale
        glm::mat4 planeModel = earthTR
            * glm::translate(glm::mat4(1.0f), orbitPos)
            * basis
            * glm::rotate(glm::mat4(1.0f), glm::radians(-45.0f), glm::vec3(0.0f, 0.0f, 1.0f))
            * glm::scale(glm::mat4(1.0f), glm::vec3(PLANE_SCALE));

        earth_shader.setMat4("model", planeModel);
        // draw with same shader uniforms (view/projection/light already bound)
        // Apply per-mesh transform to spin propeller meshes
        float propAngle = (2.0f * M_PI) * PROPELLER_RPS * elapsed_time; // radians
        spitfire_model.drawWithTransforms(earth_shader, [&](const std::string& meshName) -> glm::mat4 {
            std::string lower = meshName;
            for (char& c : lower) c = static_cast<char>(::tolower(c));
            if (lower.find("prop") != std::string::npos) {
                return glm::rotate(glm::mat4(1.0f), propAngle, PROPELLER_AXIS);
            }
            return glm::mat4(1.0f);
        }); 

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Destroy window
    glfwDestroyWindow(window);

    // Terminate and return success
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
    SCREEN_WIDTH = width;
    SCREEN_HEIGHT = height;
}
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <my_shader.h>
#include <my_model.h>
#include <my_camera.h>

#include <iostream>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Callback function declarations
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);

// Screen params
unsigned int SCREEN_WIDTH = 1920;
unsigned int SCREEN_HEIGHT = 1080;

// Timing params
float delta_time = 0.0f;
float prev_frame = 0.0f;
float elapsed_time = 0.0f;

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
const float z_pos_init = 5.0f;
Camera camera(glm::vec3(x_pos_init, y_pos_init, z_pos_init));

int setupGLFW(GLFWwindow** window) {
    // glfw init and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE); // Remove title bar

    // Screen params
    GLFWmonitor* my_monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(my_monitor);
    SCREEN_WIDTH = mode->width; 
    SCREEN_HEIGHT = mode->height;

    // glfw window creation
    GLFWwindow* glfw_window = glfwCreateWindow(
        SCREEN_WIDTH, 
        SCREEN_HEIGHT, 
        "Globe", 
        glfwGetPrimaryMonitor(), 
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

    // Disable cursor
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
    return 0;
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

    // Models
    Model earth_model(EARTH_MODEL, "Earth");
    Model spitfire_model(SPITFIRE_MODEL, "Spitfire");

    // Camera
    camera.setMouseSensitivity(mouse_sensitivity);
    camera.setCameraMovementSpeed(camera_speed);
    camera.setZoom(camera_zoom);
    camera.setFPSCamera(false, y_pos_init);
    camera.setZoomEnabled(false);

    // Render loop
    while (!glfwWindowShouldClose(window))
    {
        // Clear screen colour and buffers
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Per-frame time logic
        float current_frame = static_cast<float>(glfwGetTime());
        delta_time = current_frame - prev_frame;
        elapsed_time += delta_time;
        prev_frame = current_frame;

        // Rotate the model slowly about the y-axis
        y_rot += 20.0f * delta_time;
        y_rot = fmodf(y_rot, 360.0f);

        // Exit on ESC
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // View and projection
        camera.position = glm::vec3(0.0f, 0.0f, 8.0f);

        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.zoom),
            static_cast<float>(SCREEN_WIDTH) / static_cast<float>(SCREEN_HEIGHT), 
            0.1f, 1000.0f);

        // All have same model, view and projection 
        glm::mat4 model = glm::identity<glm::mat4>();
        // Slightly scale down to keep fully within the frame
        model = glm::scale(model, glm::vec3(0.85f));
        model = glm::rotate(model, glm::radians(y_rot), glm::vec3(0.0f, 1.0f, 0.0f));

        earth_shader.use();
        earth_shader.setMat4("model", model);
        earth_shader.setMat4("view", view);
        earth_shader.setMat4("projection", projection);

        // Subtle indoor-light uniforms
        // Lighting uniforms for current shader (world space)
        earth_shader.setVec3("lightPos", glm::vec3(-3.0f, 3.0f, 5.0f)); // from left, above, slightly forward
        earth_shader.setVec3("viewPos", camera.position);
        earth_shader.setFloat("shininess", 32.0f);

        // Draw the Earth
        earth_model.draw(earth_shader);

        // --- Draw the Spitfire orbiting the equator ---
        float theta = glm::radians(elapsed_time * PLANE_ORBIT_SPEED_DEG);
        glm::vec3 orbitPos = glm::vec3(
            PLANE_ORBIT_RADIUS * cosf(theta),
            0.0f,
            PLANE_ORBIT_RADIUS * sinf(theta)
        );

        // Tangent direction along the orbit (forward direction)
        glm::vec3 forward = glm::normalize(glm::vec3(
            -sinf(theta), 0.0f, cosf(theta)
        ));
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 right = glm::normalize(glm::cross(forward, up));
        // Recompute up to ensure orthonormal basis
        up = glm::normalize(glm::cross(right, forward));

        glm::mat4 basis(1.0f);
        // Columns are the basis vectors (world space): right, up, forward
        basis[0] = glm::vec4(right, 0.0f);
        basis[1] = glm::vec4(up, 0.0f);
        basis[2] = glm::vec4(forward, 0.0f);

        glm::mat4 planeModel = glm::identity<glm::mat4>();
        planeModel = glm::scale(planeModel, glm::vec3(PLANE_SCALE));
        planeModel = basis * planeModel;                 // rotate to face tangent
        planeModel = glm::translate(glm::mat4(1.0f), orbitPos) * planeModel; // move to orbit position

        planeModel = glm::rotate(planeModel, glm::radians(-45.0f), glm::vec3(0.0f, 0.0f, 1.0f));

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
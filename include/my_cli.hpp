#ifndef MY_CLI_HPP
#define MY_CLI_HPP

#include <glm/glm.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

struct CLIOptions {
    // Screen params
    unsigned int screenWidth{640};
    unsigned int screenHeight{480};

    // Camera params
    std::string webcamName{"Webcam"};
    std::string deviceName{"/dev/video0"};
    unsigned int fps{30};

    // ONNX yolo model params
    std::string onnxModelPath{"models/yolo11s_hand.onnx"};
    unsigned int onnxInputSize{640};
    bool applySmoothing{true};

    // Virtual camera params
    float cameraSpeed{3.0f};
    float mouseSensitivity{0.1f};
    float cameraZoom{50.0f};
    glm::vec3 initPosition{0.0f, 0.0f, 15.0f};

    // Earth model params
    std::string earthModelPath{"3d_models/earth.obj"};
    float earthScale{0.8f};

    // Moon model params
    std::string moonModelPath{"3d_models/moon.obj"};
    float moonOrbitRadius{10.0f};
    float moonOrbitSpeedDeg{30.0f};
    float moonScale{0.5f};

    // Spitfire model params
    std::string spitfireModelPath{"3d_models/spitfire.obj"};
    float spitfireOrbitRadius{4.0f};
    float spitfireOrbitSpeedDeg{60.0f};
    float spitfireScale{0.35f};
    float propellerRps{2.0f};
    glm::vec3 propellerAxis{0.0f, 0.21443f, 3.382f};

    // Shader paths
    std::string earthVertexShaderPath{"shaders/earth_shader.vs"};
    std::string earthFragmentShaderPath{"shaders/earth_shader.fs"};
    std::string bgVertexShaderPath{"shaders/bg_quad.vs"};
    std::string bgFragmentShaderPath{"shaders/bg_quad.fs"};

    // Other CLI params
    std::string configPath{"config/config.yaml"}; // Default config path
    bool show_help{false};

    // Load defaults from config.yaml
    void loadDefaults() {
        YAML::Node config = YAML::LoadFile(configPath);

        // Screen params
        if (config["screen_width"]) screenWidth = config["screen_width"].as<unsigned int>();
        if (config["screen_height"]) screenHeight = config["screen_height"].as<unsigned int>();

        // Camera params
        if (config["camera_name"]) webcamName = config["camera_name"].as<std::string>();
        if (config["device_name"]) deviceName = config["device_name"].as<std::string>();
        if (config["fps"]) fps = config["fps"].as<unsigned int>();

        // ONNX yolo model params
        if (config["onnx_input_size"]) onnxInputSize = config["onnx_input_size"].as<unsigned int>();
        if (config["apply_smoothing"]) applySmoothing = config["apply_smoothing"].as<bool>();
        if (config["model_path"]) onnxModelPath = config["model_path"].as<std::string>();

        // Virtual camera params
        if (config["camera_speed"]) cameraSpeed = config["camera_speed"].as<float>();
        if (config["mouse_sensitivity"]) mouseSensitivity = config["mouse_sensitivity"].as<float>();
        if (config["camera_zoom"]) cameraZoom = config["camera_zoom"].as<float>();
        if (config["init_position"]) {
            auto pos = config["init_position"].as<std::vector<float>>();
            if (pos.size() == 3) {
                initPosition = glm::vec3(pos[0], pos[1], pos[2]);
            }
        }

        // Earth model params
        if (config["earth_model_path"]) earthModelPath = config["earth_model_path"].as<std::string>();
        if (config["earth_scale"]) earthScale = config["earth_scale"].as<float>();

        // Moon model params
        if (config["moon_model_path"]) moonModelPath = config["moon_model_path"].as<std::string>();
        if (config["moon_orbit_radius"]) moonOrbitRadius = config["moon_orbit_radius"].as<float>();
        if (config["moon_scale"]) moonScale = config["moon_scale"].as<float>();

        // Spitfire model params
        if (config["spitfire_model_path"]) spitfireModelPath = config["spitfire_model_path"].as<std::string>();
        if (config["spitfire_orbit_radius"]) spitfireOrbitRadius = config["spitfire_orbit_radius"].as<float>();
        if (config["spitfire_orbit_speed_deg"]) spitfireOrbitSpeedDeg = config["spitfire_orbit_speed_deg"].as<float>();
        if (config["spitfire_scale"]) spitfireScale = config["spitfire_scale"].as<float>();
        if (config["propeller_rps"]) propellerRps = config["propeller_rps"].as<float>();
        if (config["propeller_axis"]) {
            auto axis = config["propeller_axis"].as<std::vector<float>>();
            if (axis.size() == 3) {
                propellerAxis = glm::vec3(axis[0], axis[1], axis[2]);
            }
        }

        // Shader paths
        if (config["earth_vertex_shader_path"]) earthVertexShaderPath = config["earth_vertex_shader_path"].as<std::string>();
        if (config["earth_fragment_shader_path"]) earthFragmentShaderPath = config["earth_fragment_shader_path"].as<std::string>();
        if (config["bg_vertex_shader_path"]) bgVertexShaderPath = config["bg_vertex_shader_path"].as<std::string>();
        if (config["bg_fragment_shader_path"]) bgFragmentShaderPath = config["bg_fragment_shader_path"].as<std::string>();
    }
};

// Parse command line arguments. Supports:
//   -h, --help
//   --screen_width <int>
//   --screen_height <int>
//   --webcam_name <string>
//   --device_name <string>
//   --fps <int>
//   --onnx_model_path <string>
//   --onnx_input_size <int>
//   --apply_smoothing <bool>
//   --camera_speed <float>
//   --mouse_sensitivity <float>
//   --camera_zoom <float>
//   --init_position <float,float,float>
//   --earth_model_path <string>
//   --earth_scale <float>
//   --moon_model_path <string>
//   --moon_orbit_radius <float>
//   --moon_scale <float>
//   --spitfire_model_path <string>
//   --spitfire_orbit_radius <float>
//   --spitfire_orbit_speed_deg <float>
//   --spitfire_scale <float>
//   --propeller_rps <float>
//   --propeller_axis <float,float,float>
//   --earth_vertex_shader_path <string>
//   --earth_fragment_shader_path <string>
//   --bg_vertex_shader_path <string>
//   --bg_fragment_shader_path <string>
//   --config_path <string> 
//   --show_help
CLIOptions parseCli(int argc, char** argv);

void printHelp(const char* prog);

#endif // MY_CLI_HPP
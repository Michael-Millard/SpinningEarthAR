#include <my_cli.hpp>

#include <iostream>
#include <vector>

static bool isFlag(const std::string& s, const char* shortf, const char* longf) {
    return s == shortf || s == longf;
}

CLIOptions parseCli(int argc, char** argv) {
    CLIOptions opts;

    std::vector<std::string> args(argv + 1, argv + argc);

    // Step 1: Extract config file path first
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (isFlag(a, "--config_path", "--config")) {
            if (i + 1 < args.size()) {
                opts.configPath = args[++i];
            } else {
                std::cerr << "Missing value for --config_path\n";
            }
            break; // Exit loop after finding config path
        }
    }

    // Step 2: Load defaults from the specified or default config path
    opts.loadDefaults();

    // Step 3: Parse remaining CLI arguments to overwrite defaults
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (isFlag(a, "-h", "--help")) {
            opts.show_help = true;
            break;
        } else if (isFlag(a, "--screen_width", "--width")) {
            if (i + 1 < args.size()) {
                try {
                    opts.screenWidth = std::stoi(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid integer for --screen_width\n";
                }
            } else {
                std::cerr << "Missing value for --screen_width\n";
            }
        } else if (isFlag(a, "--screen_height", "--height")) {
            if (i + 1 < args.size()) {
                try {
                    opts.screenHeight = std::stoi(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid integer for --screen_height\n";
                }
            } else {
                std::cerr << "Missing value for --screen_height\n";
            }
        } else if (isFlag(a, "--webcam_name", "--camera_name")) {
            if (i + 1 < args.size()) {
                opts.webcamName = args[++i];
            } else {
                std::cerr << "Missing value for " << a << "\n";
            }
        } else if (isFlag(a, "--device_name", "--device")) {
            if (i + 1 < args.size()) {
                opts.deviceName = args[++i];
            } else {
                std::cerr << "Missing value for " << a << "\n";
            }
        } else if (isFlag(a, "--FPS", "--fps")) {
            if (i + 1 < args.size()) {
                try {
                    opts.fps = std::stoi(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid integer for --FPS\n";
                }
            } else {
                std::cerr << "Missing value for --FPS\n";
            }
        } else if (isFlag(a, "--onnx_model_path", "--onnx_model")) {
            if (i + 1 < args.size()) {
                opts.onnxModelPath = args[++i];
            } else {
                std::cerr << "Missing value for --onnx_model_path\n";
            }
        } else if (isFlag(a, "--onnx_input_size", "--input_size")) {
            if (i + 1 < args.size()) {
                try {
                    opts.onnxInputSize = std::stoi(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid integer for --onnx_input_size\n";
                }
            } else {
                std::cerr << "Missing value for --onnx_input_size\n";
            }
        } else if (isFlag(a, "--apply_smoothing", "--smoothing")) {
            if (i + 1 < args.size()) {
                std::string val = args[++i];
                if (val == "true" || val == "1") {
                    opts.applySmoothing = true;
                } else if (val == "false" || val == "0") {
                    opts.applySmoothing = false;
                } else {
                    std::cerr << "Invalid value for --apply_smoothing; use true/false or 1/0\n";
                }
            } else {
                std::cerr << "Missing value for --apply_smoothing\n";
            }
        } else if (isFlag(a, "--camera_speed", "--cam_speed")) {
            if (i + 1 < args.size()) {
                try {
                    opts.cameraSpeed = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --camera_speed\n";
                }
            } else {
                std::cerr << "Missing value for --camera_speed\n";
            }
        } else if (isFlag(a, "--mouse_sensitivity", "--mouse_sens")) {
            if (i + 1 < args.size()) {
                try {
                    opts.mouseSensitivity = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --mouse_sensitivity\n";
                }
            } else {
                std::cerr << "Missing value for --mouse_sensitivity\n";
            }
        } else if (isFlag(a, "--camera_zoom", "--zoom")) {
            if (i + 1 < args.size()) {
                try {
                    opts.cameraZoom = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --camera_zoom\n";
                }
            } else {
                std::cerr << "Missing value for --camera_zoom\n";
            }
        } else if (isFlag(a, "--init_position", "--cam_pos")) {
            if (i + 1 < args.size()) {
                try {
                    opts.initPosition = glm::vec3(
                        std::stof(args[++i]),
                        std::stof(args[++i]),
                        std::stof(args[++i])
                    );
                } catch (...) {
                    std::cerr << "Invalid float for --init_position\n";
                }
            } else {
                std::cerr << "Missing value for --init_position\n";
            }
        } else if (isFlag(a, "--earth_model_path", "--earth_model")) {
            if (i + 1 < args.size()) {
                opts.earthModelPath = args[++i];
            } else {
                std::cerr << "Missing value for --earth_model_path\n";
            }
        } else if (isFlag(a, "--earth_scale", "--earth_scale")) {
            if (i + 1 < args.size()) {
                try {
                    opts.earthScale = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --earth_scale\n";
                }
            } else {
                std::cerr << "Missing value for --earth_scale\n";
            }
        } else if (isFlag(a, "--moon_model_path", "--moon_model")) {
            if (i + 1 < args.size()) {
                opts.moonModelPath = args[++i];
            } else {
                std::cerr << "Missing value for --moon_model_path\n";
            }
        } else if (isFlag(a, "--moon_orbit_radius", "--moon_radius")) {
            if (i + 1 < args.size()) {
                try {
                    opts.moonOrbitRadius = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --moon_orbit_radius\n";
                }
            } else {
                std::cerr << "Missing value for --moon_orbit_radius\n";
            }
        } else if (isFlag(a, "--moon_orbit_speed_deg", "--moon_speed")) {
            if (i + 1 < args.size()) {
                try {
                    opts.moonOrbitSpeedDeg = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --moon_orbit_speed_deg\n";
                }
            } else {
                std::cerr << "Missing value for --moon_orbit_speed_deg\n";
            }
        } else if (isFlag(a, "--moon_scale", "--moon_scale")) {
            if (i + 1 < args.size()) {
                try {
                    opts.moonScale = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --moon_scale\n";
                }
            } else {
                std::cerr << "Missing value for --moon_scale\n";
            }
        } else if (isFlag(a, "--spitfire_model_path", "--spitfire_model")) {
            if (i + 1 < args.size()) {
                opts.spitfireModelPath = args[++i];
            } else {
                std::cerr << "Missing value for --spitfire_model_path\n";
            }
        } else if (isFlag(a, "--spitfire_orbit_radius", "--spitfire_radius")) {
            if (i + 1 < args.size()) {
                try {
                    opts.spitfireOrbitRadius = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --spitfire_orbit_radius\n";
                }
            } else {
                std::cerr << "Missing value for --spitfire_orbit_radius\n";
            }
        } else if (isFlag(a, "--spitfire_orbit_speed_deg", "--spitfire_speed")) {
            if (i + 1 < args.size()) {
                try {
                    opts.spitfireOrbitSpeedDeg = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --spitfire_orbit_speed_deg\n";
                }
            } else {
                std::cerr << "Missing value for --spitfire_orbit_speed_deg\n";
            }
        } else if (isFlag(a, "--spitfire_scale", "--spitfire_scale")) {
            if (i + 1 < args.size()) {
                try {
                    opts.spitfireScale = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --spitfire_scale\n";
                }
            } else {
                std::cerr << "Missing value for --spitfire_scale\n";
            }
        } else if (isFlag(a, "--propeller_rps", "--prop_rps")) {
            if (i + 1 < args.size()) {
                try {
                    opts.propellerRps = std::stof(args[++i]);
                } catch (...) {
                    std::cerr << "Invalid float for --propeller_rps\n";
                }
            } else {
                std::cerr << "Missing value for --propeller_rps\n";
            }
        } else if (isFlag(a, "--propeller_axis", "--prop_axis")) {
            if (i + 1 < args.size()) {
                try {
                    opts.propellerAxis = glm::vec3(
                        std::stof(args[++i]),
                        std::stof(args[++i]),
                        std::stof(args[++i])
                    );
                } catch (...) {
                    std::cerr << "Invalid float for --propeller_axis\n";
                }
            } else {
                std::cerr << "Missing value for --propeller_axis\n";
            }
        } else if (isFlag(a, "--config_path", "--config")) {
            if (i + 1 < args.size()) {
                opts.configPath = args[++i];
            } else {
                std::cerr << "Missing value for --config_path\n";
            }
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
        }
    }

    return opts;
}

void printHelp(const char* prog) {
    std::cout << "Usage: " << (prog ? prog : "hand-detector")
        << " [options]" << '\n'
        << "\n"
        << "Options:\n"
        << "  --webcam_name <string>                    Name of the webcam (default: Webcam)\n"
        << "  --device_name <string>                    Device name (default: /dev/video0)\n"
        << "  --screen_width <int>                      Screen width (default: 640)\n"
        << "  --screen_height <int>                     Screen height (default: 480)\n"
        << "  --fps <int>                               Frames per second (default: 60)\n"
        << "  --onnx_model_path <string>                Path to ONNX model (default: models/yolo11s_hand.onnx)\n"
        << "  --onnx_input_size <int>                   ONNX model input size (default: 640)\n"
        << "  --apply_smoothing <bool>                  Apply smoothing to hand tracking (default: true)\n"
        << "  --camera_speed <float>                    Camera movement speed (default: 1.0)\n"
        << "  --mouse_sensitivity <float>               Mouse sensitivity (default: 0.1)\n"
        << "  --camera_zoom <float>                     Camera zoom level (default: 45.0)\n"
        << "  --init_position <float,float,float>       Initial camera position (default: 0.0,0.0,3.0)\n"
        << "  --earth_model_path <string>               Path to Earth model (default: models/earth.obj)\n"
        << "  --earth_scale <float>                     Scale of the Earth model (default: 1.0)\n"
        << "  --spitfire_model_path <string>            Path to Spitfire model (default: models/spitfire.obj)\n"
        << "  --spitfire_orbit_radius <float>           Orbit radius of Spitfire (default: 5.0)\n"
        << "  --spitfire_orbit_speed_deg <float>        Orbit speed of Spitfire in degrees per second (default: 30.0)\n"
        << "  --spitfire_scale <float>                  Scale of the Spitfire model (default: 0.5)\n"
        << "  --propeller_rps <float>                   Rotations per second of the propeller (default: 10.0)\n"
        << "  --propeller_axis <float,float,float>      Axis of propeller rotation (default: 0.0,1.0,0.0)\n"
        << "  --config_path <string>                    Path to configuration file (default: config/config.yaml)\n"
        << "  -h, --help                                Show this help message and exit\n"
        << std::endl;
}

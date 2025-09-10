cmake_minimum_required(VERSION 3.10)
project(MillSpinningGlobe LANGUAGES CXX C)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wall)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Add debug symbols for Debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

# --- Dependencies ---
find_package(PkgConfig REQUIRED)

# GLFW (via pkg-config)
pkg_search_module(GLFW REQUIRED glfw3 IMPORTED_TARGET)

# Assimp
find_package(assimp REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# OpenCV
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)

# YAML-CPP
find_package(yaml-cpp REQUIRED)

# --- Executable ---
add_executable(MillSpinningGlobe)

# Add sources explicitly
target_sources(MillSpinningGlobe PRIVATE
    main.cpp
    src/glad.c
    src/stb.cpp  
    src/my_webcam.cpp
    src/my_hands.cpp
    src/my_cli.cpp
    src/my_bg_quad.cpp
)

# Project includes (your local include/ with glad/)
target_include_directories(MillSpinningGlobe PRIVATE
    ${GLFW_INCLUDE_DIRS}
    ${ASSIMP_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Linking ---
target_link_libraries(MillSpinningGlobe PRIVATE
    PkgConfig::GLFW
    assimp
    OpenGL::GL
    ${OpenCV_LIBS}
    yaml-cpp
    dl   # required for glad on Linux
)


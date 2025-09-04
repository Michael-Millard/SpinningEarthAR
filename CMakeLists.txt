cmake_minimum_required(VERSION 3.10)
project(MillSpinningGlobe LANGUAGES CXX C)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---
find_package(PkgConfig REQUIRED)

# GLFW (via pkg-config)
pkg_search_module(GLFW REQUIRED glfw3 IMPORTED_TARGET)

# Assimp
find_package(assimp REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# --- Executable ---
add_executable(MillSpinningGlobe)

# Add sources explicitly
target_sources(MillSpinningGlobe PRIVATE
    main.cpp
    src/glad.c
    src/stb.cpp  
)

# Project includes (your local include/ with glad/)
target_include_directories(MillSpinningGlobe PRIVATE
    ${GLFW_INCLUDE_DIRS}
    ${ASSIMP_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Linking ---
target_link_libraries(MillSpinningGlobe PRIVATE
    PkgConfig::GLFW
    assimp
    OpenGL::GL
    dl   # required for glad on Linux
)


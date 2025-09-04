# Simple Makefile wrapper for CMake build

.PHONY: all build run clean reconfigure

BUILD_DIR := build
TARGET := MillSpinningGlobe

all: build

build:
	@cmake -S . -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR) -j

run: build
	@./$(BUILD_DIR)/$(TARGET)

clean:
	@rm -rf $(BUILD_DIR)

reconfigure:
	@rm -rf $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR) -j


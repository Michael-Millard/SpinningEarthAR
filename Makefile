PROJECT_NAME := MillSpinningGlobe
BUILD_DIR := build
ARGS ?= "" # Can modify to add some default args

.PHONY: all debug release clean run help

all: release

$(BUILD_DIR):
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

release: $(BUILD_DIR)
	cmake --build $(BUILD_DIR) --config Release

debug:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(BUILD_DIR) --config Debug

clean:
	rm -rf $(BUILD_DIR)

run: release
	./$(BUILD_DIR)/$(PROJECT_NAME) $(ARGS)

help:
	@echo "Targets:"
	@echo "  make            -> same as 'make release'"
	@echo "  make release    -> build Release"
	@echo "  make debug      -> build Debug"
	@echo "  make run        -> build then run with defaults or provided vars"
	@echo "  make clean      -> remove build/"
	@echo ""

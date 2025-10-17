CXX = g++
CXX_STANDARD = c++14
CXXFLAGS = -std=$(CXX_STANDARD) -g
INCLUDES = -Iinclude
SRC_DIR = src
BUILD_DIR = build

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
MAIN_OBJ = $(BUILD_DIR)/complentaryFilterMain.o

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile main file to object file
$(MAIN_OBJ): complentaryFilterMain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link all object files to create executable
complemntaryFilter: $(OBJECTS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o complmentary.out

clean:
	rm -rf $(BUILD_DIR) *.out

.PHONY: complemntaryFilter clean

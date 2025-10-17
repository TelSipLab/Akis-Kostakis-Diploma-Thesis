CXX = g++
CXX_STANDARD = c++14
CXXFLAGS = -std=$(CXX_STANDARD) -g
INCLUDES = -Iinclude
SRC_DIR = src
BUILD_DIR = build

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
COMPLEMENTARY_MAIN_OBJ = $(BUILD_DIR)/complentaryFilterMain.o
EKF_MAIN_OBJ = $(BUILD_DIR)/ekfFilter.o

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile complementary filter main file to object file
$(COMPLEMENTARY_MAIN_OBJ): complentaryFilterMain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile EKF filter main file to object file
$(EKF_MAIN_OBJ): ekfFilter.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link all object files to create complementary filter executable
complemntaryFilter: $(OBJECTS) $(COMPLEMENTARY_MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o complmentary.out

# Link all object files to create EKF filter executable
ekfFilter: $(OBJECTS) $(EKF_MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o ekf.out

clean:
	rm -rf $(BUILD_DIR) *.out

.PHONY: complemntaryFilter ekfFilter clean

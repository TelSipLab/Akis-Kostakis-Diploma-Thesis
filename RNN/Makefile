# Compiler settings
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra
LDFLAGS :=

# LibTorch path (modify this to your libtorch installation path)
LIBTORCH_PATH := /home/kostakis/dev/libtorch

# LibTorch flags
TORCH_CXX_FLAGS := -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_INCLUDE := -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
TORCH_LIBS := -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10
TORCH_LIBS += -Wl,-rpath,$(LIBTORCH_PATH)/lib

# If using CUDA, uncomment the following:
# TORCH_LIBS += -ltorch_cuda -lc10_cuda

# Target executable
TARGET := example
SOURCES := mainExample.cpp
OBJECTS := $(SOURCES:.cpp=.o)

# Build type (Release or Debug)
BUILD_TYPE ?= Release

ifeq ($(BUILD_TYPE),Release)
    CXXFLAGS += -O3 -DNDEBUG
else
    CXXFLAGS += -g -O0
endif

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS) $(TORCH_LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_CXX_FLAGS) $(TORCH_INCLUDE) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild
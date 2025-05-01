cmake_minimum_required(VERSION 3.16)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Precompiled header
set(PCH_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/pch.h")

# Collect all .cpp files
file(GLOB SOURCES "*.cpp")

# Define the executable
add_executable(main.out ${SOURCES})

# Use precompiled header
target_precompile_headers(main.out PRIVATE ${PCH_HEADER})

# Optional: Enable debug symbols
target_compile_options(main.out PRIVATE -g)

# Optional: Add 'make run'
add_custom_target(run
    COMMAND main.out
    DEPENDS main.out
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

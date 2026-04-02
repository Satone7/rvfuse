cmake_minimum_required(VERSION 3.16)
project(yolo_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static linking
set(BUILD_SHARED_LIBS OFF)

# Add executable
add_executable(yolo_inference
    yolo_runner.cpp
)

# Include directories
target_include_directories(yolo_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE}
)

# Link libraries
target_link_libraries(yolo_inference PRIVATE
    ${ONNXRUNTIME_LIB}
    pthread
)

# Compiler flags for debug info
target_compile_options(yolo_inference PRIVATE -O2 -g)

# Linker flags for static build
target_link_options(yolo_inference PRIVATE -static)

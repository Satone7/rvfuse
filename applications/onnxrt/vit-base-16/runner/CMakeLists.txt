cmake_minimum_required(VERSION 3.16)
project(vit_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ORT install directory (passed via -DORT_INSTALL_DIR=...)
if(NOT DEFINED ORT_INSTALL_DIR)
    message(FATAL_ERROR "ORT_INSTALL_DIR is not set. Pass -DORT_INSTALL_DIR=<path>")
endif()

# Include ORT headers — installed under include/onnxruntime/
include_directories(${ORT_INSTALL_DIR}/include/onnxruntime)

# stb_image.h from YOLO runner (shared header-only library)
set(STB_IMAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../yolo/runner")
if(EXISTS "${STB_IMAGE_DIR}/stb_image.h")
    include_directories(${STB_IMAGE_DIR})
else()
    message(FATAL_ERROR "stb_image.h not found at ${STB_IMAGE_DIR}")
endif()

# Build the runner
add_executable(vit_inference
    vit_runner.cpp
)

# Link against ORT shared library
target_link_directories(vit_inference PRIVATE
    ${ORT_INSTALL_DIR}/lib
)

target_link_libraries(vit_inference PRIVATE
    onnxruntime
)

# Strip is OFF by default for profiling (debug symbols preserved)
# Enable strip for deployment: cmake -DSTRIP_BINARY=ON
if(STRIP_BINARY)
    add_custom_command(TARGET vit_inference POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:vit_inference>
        COMMENT "Stripping debug symbols from vit_inference"
    )
endif()

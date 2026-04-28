cmake_minimum_required(VERSION 3.16)
project(superglue_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ORT install directory (passed via -DORT_INSTALL_DIR=...)
if(NOT DEFINED ORT_INSTALL_DIR)
    message(FATAL_ERROR "ORT_INSTALL_DIR is not set. Pass -DORT_INSTALL_DIR=<path>")
endif()

# Include ORT headers — installed under include/onnxruntime/
include_directories(${ORT_INSTALL_DIR}/include/onnxruntime)

# Build the runner (no stb_image needed — SuperGlue uses synthetic/file data)
add_executable(superglue_inference
    superglue_runner.cpp
)

# Link against ORT shared library
target_link_directories(superglue_inference PRIVATE
    ${ORT_INSTALL_DIR}/lib
)

target_link_libraries(superglue_inference PRIVATE
    onnxruntime
)

# Strip is OFF by default for profiling (debug symbols preserved)
if(STRIP_BINARY)
    add_custom_command(TARGET superglue_inference POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:superglue_inference>
        COMMENT "Stripping debug symbols from superglue_inference"
    )
endif()

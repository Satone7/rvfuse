cmake_minimum_required(VERSION 3.14)
project(KCFcpp)

set(CMAKE_CXX_STANDARD 17)

# Use the parent toolchain
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../riscv64-linux-toolchain.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../riscv64-linux-toolchain.cmake" CACHE STRING "" FORCE)
endif()

find_package(OpenCV REQUIRED)

# KCF tracker library (without main)
include_directories(${OpenCV_INCLUDE_DIRS} vendor/KCFcpp/src)
set(KCF_TRACKER_SOURCES
    vendor/KCFcpp/src/kcftracker.cpp
    vendor/KCFcpp/src/fhog.cpp
)
add_library(kcf_tracker STATIC ${KCF_TRACKER_SOURCES})
target_link_libraries(kcf_tracker ${OpenCV_LIBS})
target_compile_options(kcf_tracker PRIVATE -O2 -w)

# Video runner (for video input)
add_executable(kcf_video runner/kcf_video.cpp)
target_link_libraries(kcf_video kcf_tracker ${OpenCV_LIBS})

# Original image sequence runner (optional)
FILE(GLOB_RECURSE sourcefiles "vendor/KCFcpp/src/*.cpp")
add_executable(KCF ${sourcefiles})
target_link_libraries(KCF ${OpenCV_LIBS})
target_compile_options(KCF PRIVATE -O2 -w)
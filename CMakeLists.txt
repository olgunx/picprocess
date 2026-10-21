cmake_minimum_required(VERSION 3.10)
project(fastcrop C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -pthread)
    
    # Raspberry Pi Zero 2 ARM tuning
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        add_compile_options(-march=armv8-a+crc -mtune=cortex-a53)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7.*|armhf")
        add_compile_options(-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard)
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

# Try standard FindJPEG first
find_package(JPEG QUIET)

if(NOT JPEG_FOUND)
    # Fallback to local include directory and system libjpeg runtime library
    find_library(JPEG_RUNTIME_LIB NAMES jpeg libjpeg.so.8 libjpeg.so.62 libjpeg.so
        PATHS /usr/lib /usr/local/lib /lib /usr/lib/aarch64-linux-gnu /usr/lib/arm-linux-gnueabihf /usr/lib/x86_64-linux-gnu)
    
    if(JPEG_RUNTIME_LIB)
        message(STATUS "Found libjpeg runtime: ${JPEG_RUNTIME_LIB}")
        set(JPEG_LIBRARIES ${JPEG_RUNTIME_LIB})
    else()
        message(FATAL_ERROR "Could not find libjpeg library. Install libjpeg-dev with: sudo apt install libjpeg-dev")
    endif()
else()
    set(JPEG_LIBRARIES JPEG::JPEG)
endif()

add_executable(fastcrop src/fastcrop.c)
add_executable(fastdetect src/fastdetect.c)

target_link_libraries(fastcrop PRIVATE
    ${JPEG_LIBRARIES}
    Threads::Threads
    m
)

target_link_libraries(fastdetect PRIVATE
    ${JPEG_LIBRARIES}
    Threads::Threads
    m
)

install(TARGETS fastcrop fastdetect DESTINATION bin)
install(PROGRAMS fastcrop.py DESTINATION bin RENAME fastcrop-py)
install(PROGRAMS fastdetect.py DESTINATION bin RENAME fastdetect-py)

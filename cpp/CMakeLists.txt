cmake_minimum_required(VERSION 3.18)
project(arcvision_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(TRT_ROOT "" CACHE PATH "TensorRT installation root")
if(NOT TRT_ROOT)
    if(DEFINED ENV{TRT_ROOT})
        set(TRT_ROOT $ENV{TRT_ROOT})
    else()
        set(TRT_ROOT "/usr/local/tensorrt")
    endif()
endif()

find_path(TRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${TRT_ROOT}/include
)

find_library(TRT_LIBRARY
    NAMES nvinfer
    HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64
)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIBRARY)
    message(WARNING "TensorRT not found. Set TRT_ROOT cmake variable or environment variable.")
endif()

find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ libzmq)
endif()

if(NOT ZMQ_FOUND)
    find_path(ZMQ_INCLUDE_DIR NAMES zmq.h)
    find_library(ZMQ_LIBRARY NAMES zmq)
    if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
        set(ZMQ_FOUND TRUE)
    endif()
endif()

set(SOURCES
    src/shm_frame_queue.cpp
    src/trt_engine.cpp
    src/arc_detector.cpp
    src/zmq_publisher.cpp
    src/engine_main.cpp
)

set(CUDA_SOURCES
    src/cuda_preprocess.cu
)

add_library(arcvision_core SHARED
    ${SOURCES}
    ${CUDA_SOURCES}
)

target_include_directories(arcvision_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
        ${TRT_INCLUDE_DIR}
        ${ZMQ_INCLUDE_DIR}
)

target_link_libraries(arcvision_core
    PUBLIC
        ${TRT_LIBRARY}
        ${CUDAToolkit_LIBRARIES}
        cuda
        cudart
        ${ZMQ_LIBRARY}
)

if(WIN32)
    target_compile_definitions(arcvision_core PRIVATE _WIN32)
else()
    target_compile_definitions(arcvision_core PRIVATE __linux__)
endif()

target_compile_options(arcvision_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(arcvision_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
)

install(TARGETS arcvision_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

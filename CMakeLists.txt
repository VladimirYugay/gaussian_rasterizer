cmake_minimum_required(VERSION 3.18)
project(rasterizer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CMAKE_PREFIX_PATH /home/vy/projects/libtorch)

find_package(Python REQUIRED Development)
find_package(Torch REQUIRED)

set(CUDA_SOURCES
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.cu
    cuda_rasterizer/rasterizer_impl.cu
    rasterize_points.cu)
add_executable(rasterizer main.cpp ${CUDA_SOURCES})

set_target_properties(rasterizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(rasterizer PRIVATE TORCH_USE_CUDA_DSA)
target_link_libraries(rasterizer "${TORCH_LIBRARIES}")
target_link_libraries(rasterizer Python::Python)
target_include_directories(rasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_compile_options(rasterizer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)

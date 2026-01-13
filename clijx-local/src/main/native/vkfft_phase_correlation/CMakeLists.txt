cmake_minimum_required(VERSION 3.10)
project(clijx_vkfft_phase_correlation CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(JNI REQUIRED)

add_library(clijx_vkfft SHARED vkfft_phase_correlation.cpp)

target_include_directories(clijx_vkfft
  PRIVATE
    ${OpenCL_INCLUDE_DIRS}
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/VkFFT/vkFFT
)

target_compile_definitions(clijx_vkfft PRIVATE VKFFT_BACKEND=3)
target_link_libraries(clijx_vkfft PRIVATE ${OpenCL_LIBRARIES})

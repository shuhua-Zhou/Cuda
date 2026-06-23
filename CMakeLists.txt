cmake_minimum_required(VERSION 3.18)

project(CudaOpenCvImagePipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cuda_image_pipeline
    src/cuda_image_pipeline.cu
)

target_include_directories(cuda_image_pipeline PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cuda_image_pipeline PRIVATE
    ${OpenCV_LIBS}
)

set_target_properties(cuda_image_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

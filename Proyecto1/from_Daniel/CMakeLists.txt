cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
#CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)
set(LTI_CONFIG_CMD /home/alvaro/Documents/Maestria/PDI/ltilib-2/linux/lti-local-config)
project(cmake_and_cuda LANGUAGES CXX CUDA)

find_package(lti CONFIG REQUIRED HINTS .)

# Create real project
add_executable(project
    project.hpp
    project.cpp
    project.cu
)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
target_compile_features(project PUBLIC cxx_std_11)

target_link_libraries(project PUBLIC lti)

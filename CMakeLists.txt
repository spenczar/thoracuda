cmake_minimum_required(VERSION 3.10)

project(thoracuda
  VERSION 0.1.0
  DESCRIPTION "CUDA-accelerated implementation of THOR, an orbit recovery algorithm for small bodies in the solar system"
  LANGUAGES CXX CUDA
)

# thoracuda: the library
add_library(thoracuda
  src/cpp/cartesian_point_sources.cpp
  src/cpp/gnomonic_point_sources.cpp    
  src/cpp/projections.cpp
)
target_compile_features(thoracuda PUBLIC cxx_std_17)
target_include_directories(thoracuda PUBLIC extern/eigen src/cpp)

add_library(thoracuda_cuda
  src/cuda/rangearray.cu
)
target_include_directories(thoracuda_cuda PUBLIC src/cuda)

# thoracuda-cli: the executable
add_executable(thoracuda-cli
  src/cpp/bin/thoracuda.cpp
)
target_link_libraries(thoracuda-cli thoracuda thoracuda_cuda)

# tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

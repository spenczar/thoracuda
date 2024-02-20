cmake_minimum_required(VERSION 3.10)

# If debug:
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_CXX_FLAGS "-g -fno-omit-frame-pointer -Wall -Wextra -Wpedantic -Werror")
	set(CMAKE_CUDA_FLAGS "-lineinfo")
endif()
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
  src/cpp/kdtree.cpp
  src/cpp/dbscan.cpp
  src/cpp/stats.cpp
)
target_compile_features(thoracuda PUBLIC cxx_std_17)
target_include_directories(thoracuda PUBLIC extern/eigen src/cpp)

add_library(thoracuda_cuda
  src/cuda/rangearray.cu
  src/cuda/range_brute.cu
  src/cuda/pairminmax.cu
  src/cuda/hotspot2d.cu
)
target_include_directories(thoracuda_cuda PUBLIC src/cuda)
# Architecture: Pascal and later, since I have a GTX 1080 at home.
set_target_properties(thoracuda_cuda PROPERTIES CUDA_ARCHITECTURES 61)

# thoracuda-cli: the executable
add_executable(thoracuda-cli
  src/cpp/bin/thoracuda.cpp
)
target_link_libraries(thoracuda-cli thoracuda thoracuda_cuda)

add_executable(thoracuda-pminmax
  src/cpp/bin/thoracuda_pminmax.cpp
)
target_link_libraries(thoracuda-pminmax thoracuda thoracuda_cuda)

# tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

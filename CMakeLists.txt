cmake_minimum_required(VERSION 3.10)

project(thoracuda
  VERSION 0.1.0
  DESCRIPTION "CUDA-accelerated implementation of THOR, an orbit recovery algorithm for small bodies in the solar system"
  LANGUAGES CXX CUDA
)

add_executable(thoracuda
  src/cpp/main.cpp
)

#pragma once

// This file contains some useful macros for CUDA programming.
//
// PRINT_CUDA_ERRORS: if set to 1, print out CUDA errors to stderr.
#ifndef PRINT_CUDA_ERRORS
#define PRINT_CUDA_ERRORS 1
#endif

#if PRINT_CUDA_ERRORS
#define PRINT_CUDA_ERROR(err)                                            \
  do {                                                                   \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
  } while (0)
#else
#define PRINT_CUDA_ERROR(err) \
  do {                        \
  } while (0)
#endif

// CUDA error macros
// =================
//
// These have a few requirements:
//
// 1. The code must be in a function that has a label "fail" that is
//    used to clean up resources and return an error code.
//
// 2. The error code must be of type cudaError_t, or int.

// CUDA_OR_FAIL: if the expression expr does not return cudaSuccess,
// print out the error and jump to the label fail.
#define CUDA_OR_FAIL(expr)    \
  do {                        \
    err = (expr);             \
    if (err != cudaSuccess) { \
      PRINT_CUDA_ERROR(err);  \
      goto fail;              \
    }                         \
  } while (0)

// CUDA_CHECK_ERROR: check if there is an outstanding CUDA error, and if so,
// print it out and jump to the label fail.
#define CUDA_CHECK_ERROR()    \
  do {                        \
    err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      PRINT_CUDA_ERROR(err);  \
      goto fail;              \
    }                         \
  } while (0)

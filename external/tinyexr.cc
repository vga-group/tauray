#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_THREAD 1
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcalloc-transposed-args"
#endif
#include "tinyexr.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

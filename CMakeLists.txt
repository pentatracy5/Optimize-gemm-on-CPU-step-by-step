cmake_minimum_required(VERSION 3.18)

project(OptimizeGemm
    LANGUAGES CXX
    VERSION 1.0.0
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Make project-local Find modules available.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Global dependencies used by the library and/or the executable.
find_package(OpenMP REQUIRED)

# Intel MKL: use the single dynamic library (mkl_rt) with Intel OpenMP threading.
find_package(MKL REQUIRED)

add_subdirectory(src)

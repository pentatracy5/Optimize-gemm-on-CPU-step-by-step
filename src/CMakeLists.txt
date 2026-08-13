# Core GEMM library: matmul implementations + helper utilities.
add_library(matmul_lib STATIC
    matmul.cpp
    utils.cpp
)

target_include_directories(matmul_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# OpenMP is used directly in matmul.cpp and config.h.
target_link_libraries(matmul_lib PUBLIC OpenMP::OpenMP_CXX)

# Intel MKL (cblas_sgemm is used in the reference implementation).
target_link_libraries(matmul_lib PUBLIC MKL::MKL)

# Driver executable.
add_executable(matmul main.cpp)
target_link_libraries(matmul PRIVATE matmul_lib)

# Release-only optimization flags.
if(MSVC)
    target_compile_options(matmul_lib PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX2>)
    target_compile_options(matmul PRIVATE $<$<CONFIG:Release>:/O2 /arch:AVX2>)
else()
    target_compile_options(matmul_lib PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
    target_compile_options(matmul PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
endif()

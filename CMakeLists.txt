cmake_minimum_required(VERSION 3.22)
project(diffcg_binaries LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Pre-flight: check for nvcc before project() enables CUDA language
find_program(NVCC_PROGRAM nvcc)
if(NOT NVCC_PROGRAM)
    message(FATAL_ERROR
        "CUDA Toolkit not found — nvcc is not on PATH.\n"
        "diffcg requires CUDA >= 11.0 with nvcc on PATH.\n"
        "Install CUDA from: https://developer.nvidia.com/cuda-downloads\n"
        "If CUDA is installed, add it to PATH, e.g.:\n"
        "  export PATH=/usr/local/cuda/bin:\$PATH"
    )
endif()
message(STATUS "Found nvcc: ${NVCC_PROGRAM}")

find_package(CUDAToolkit REQUIRED)

# Global CUDA flags for fastMD
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -DNDEBUG")

# ---- GPUMD ----
# Built via its own makefile; we drive it through add_custom_target
# Override the hardcoded -arch=sm_60 with the CMAKE_CUDA_ARCHITECTURES value
set(GPUMD_ARCH_FLAG "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}")

add_custom_target(gpumd_build ALL
    COMMAND make -j CC=${CMAKE_CUDA_COMPILER} CUDA_ARCH="${GPUMD_ARCH_FLAG}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vendor/gpumd
    COMMENT "Building GPUMD and NEP..."
)

# ---- fastMD ----
add_subdirectory(vendor/fastmd)

# ---- Install ----
install(PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/gpumd/gpumd
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/gpumd/nep
    DESTINATION diffcg/bin
)
install(TARGETS fastmd DESTINATION diffcg/bin)

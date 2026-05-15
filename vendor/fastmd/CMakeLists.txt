# Core library (built once, shared by fastMD executable)
add_library(core STATIC
    src/core/system.cu
    src/core/morton.cu
    src/neighbor/tile_list.cu
    src/neighbor/verlet_list.cu
    src/force/lj.cu
    src/force/table.cu
    src/force/fene.cu
    src/force/angle.cu
    src/integrate/langevin.cu
    src/integrate/nose_hoover.cu
    src/analysis/thermo.cu
    src/analysis/rg.cu
    src/analysis/correlator.cu
    src/io/dump.cu
    src/io/lammps_data.cpp
    src/io/table_parser.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Main executable — lowercase output name so _binaries.py finds "fastmd"
add_executable(fastmd src/main.cu src/io/config.cpp)
target_include_directories(fastmd PRIVATE src)
target_link_libraries(fastmd core CUDA::cudart CUDA::curand)
set_target_properties(fastmd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME fastmd
)

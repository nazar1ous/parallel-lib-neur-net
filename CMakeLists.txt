cmake_minimum_required(VERSION 3.15)
project(parallel_neural_net_lib
        VERSION 1.0.0
        DESCRIPTION "Library"
        HOMEPAGE_URL "https://github.com/nazar1ous/parallel-lib-neur-net"
        LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Options
option(WARNINGS_AS_ERRORS   "Treat compiler warnings as errors."    ON)
option(ENABLE_CONAN         "Use Conan as a package manager."       OFF)
option(ENABLE_PVS_STUDIO    "Use PVS-Studio static code analyzer."  OFF) # Option for the local usage only. PVS-Studio isn't installed on GitHub machines.
option(ENABLE_SANITIZERS    "Use sanitizers to detect errors."      OFF) # Option for the test builds. Do not use it in the production builds.

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.71.0 COMPONENTS program_options mpi serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(OpenMP REQUIRED)

# Define ALL_TARGETS variable to use in some directives

# Project source compilation
include_directories(inc/eigen-3.3.9 inc src)

add_library(liba_mpi_dnn STATIC inc/layers/config.h
        inc/mpi/loss_mpi.h
        inc/mpi/fc_layer_mpi.h inc/mpi/optimizers_mpi.h
        inc/mpi/acivations_mpi.h inc/mpi/serialization_config.h src/models/dnn_model.cpp
        inc/process_data/process_data.h inc/time_counting/time_counting.h)


add_library(liba_single_process_dnn STATIC inc/layers/fc_layer.h src/layers/fc_layer.cpp inc/layers/activations.h
        inc/layers/optimizers.h inc/models/dnn_model.h src/models/dnn_model.cpp inc/layers/config.h
        inc/layers/loss.h src/models/dnn_model.cpp inc/process_data/process_data.h)

add_executable(dnn_model_single_process src/main_single_process.cpp)
add_executable(dnn_model_data_parallel src/main_data_parallel.cpp)

add_executable(dnn_model_mpi src/main_tutorial_mpi.cpp)

target_link_libraries(dnn_model_single_process liba_single_process_dnn)
target_link_libraries(dnn_model_data_parallel liba_single_process_dnn)
target_link_libraries(dnn_model_mpi liba_mpi_dnn)


target_link_libraries(liba_mpi_dnn Boost::program_options Boost::mpi Boost::serialization)
target_link_libraries(liba_mpi_dnn ${Boost_LIBRARIES})
target_link_libraries(liba_mpi_dnn ${MPI_C_LIBRARIES})
target_link_libraries(liba_mpi_dnn MPI::MPI_CXX)
target_link_libraries(liba_mpi_dnn OpenMP::OpenMP_CXX)

target_link_libraries(dnn_model_mpi Boost::program_options Boost::mpi Boost::serialization)
target_link_libraries(dnn_model_mpi ${Boost_LIBRARIES})
target_link_libraries(dnn_model_mpi ${MPI_C_LIBRARIES})
target_link_libraries(dnn_model_mpi MPI::MPI_CXX)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(ALL_TARGETS liba_mpi_dnn dnn_model_mpi dnn_model_single_process dnn_model_mpi dnn_model_data_parallel)




find_package(MPI REQUIRED) #make it REQUIRED, if you want
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


# Include default CMake configuration
include(cmake/Config.cmake)
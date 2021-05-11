cmake_minimum_required(VERSION 3.15)
project(parallel_neural_net_lib
        VERSION 1.0.0
        DESCRIPTION "Library"
        HOMEPAGE_URL "https://github.com/nazar1ous/parallel-lib-neur-net"
        LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Options
option(WARNINGS_AS_ERRORS   "Treat compiler warnings as errors."    ON)
option(ENABLE_CONAN         "Use Conan as a package manager."       OFF)
option(ENABLE_PVS_STUDIO    "Use PVS-Studio static code analyzer."  OFF) # Option for the local usage only. PVS-Studio isn't installed on GitHub machines.
option(ENABLE_SANITIZERS    "Use sanitizers to detect errors."      OFF) # Option for the test builds. Do not use it in the production builds.

# Project source compilation
include_directories(inc/eigen-3.3.9 inc src)
add_library(liba STATIC inc/layers/fc_layer.h src/layers/fc_layer.cpp inc/layers/activations.h
        inc/layers/optimizers.h inc/models/dnn_model.h src/models/dnn_model.cpp inc/layers/config.h
        inc/layers/loss.h)


add_executable(parallel_neural_net_lib src/main.cpp)
target_link_libraries(parallel_neural_net_lib liba)





# Add external packages
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.71.0 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
target_link_libraries(parallel_neural_net_lib Boost::program_options)

# Define ALL_TARGETS variable to use in some directives
set(ALL_TARGETS liba parallel_neural_net_lib)

# Include default CMake configuration
include(cmake/Config.cmake)

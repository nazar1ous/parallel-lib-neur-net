# Parallel neural network library
## Team

 - [Nazarii Kuspys](https://github.com/nazar1ous)
 - [Dmytro Lutchyn](https://github.com/dlutchyn)


## Usage
Python script automatically builds project

## Prerequisites

 - **C++ compiler** - needs to support **C++17** standard
 - **CMake** 3.15+
 - **boost MPI**
 - **openMP**
 - **eigen3.3.9**
 
## Installing

1. Clone the project.
    ```bash
    git clone https://github.com/nazar1ous/parallel-lib-neur-net
    ```
2. Install required packages.
3. Build.
    ```bash
    cmake -Bbuild
    cmake --build build
    ```

## Tutotials
# Important for MPI -- model parallelism
```main_tutorial_mpi.cpp```
To run learning model with MPI with N layers we need to specify exactly N processes for MPI.
Also, we need to cache somehow model, and then we run learning.
You should specify some directory, so that DNN_model_MPI can save information of layer weights.
1. After building in build/ directory.
    ```bash
    cd .. && rm -rf build && cmake -Bbuild && cmake --build build && cd build && mpirun --oversubscribe -np <N> ./dnn_model_mpi
    ```
2. To estimate the accuracy we need to renew model using learnt weights of the previous stage.
    Just run the same .cpp file without using MPI, and commenting the MPI related code.
 
# Important for OpenMP -- data parallelism
```main_data_parallel.cpp```
Specify the correct number of threads.

# Usual usage of DNN with FC layers
```main_single_process.cpp```

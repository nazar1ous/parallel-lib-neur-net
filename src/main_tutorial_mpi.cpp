#include <iostream>
#include "layers/config.h"
#include "mpi/dnn_model_cached.h"
#include "process_data/process_data.h"


void test_algorithm(int argc, char *argv[]){
    mpi::environment env(argc, argv);
    mpi::communicator world;
    bool is_generator = world.rank() > 0;
    mpi::communicator slaves = world.split(is_generator? 1: 0);
    md m;
    FCLayer layer;
    int epochs;
    std::vector<FCLayer> layers;
    int rank = world.rank();
    int N = world.size();
    int inp_data;
    std::vector<int> v_temp(N);

    bool stage_identifier_forward = true;
    if (world.rank() == 0){
        int x_start = 228;
        int y_end = 1488;
        v_temp[1] = x_start; v_temp[N-1] = y_end;
        mpi::scatter(world, v_temp, inp_data, 0);
    }else{

        int x;
        int y;
        int dA;
        int Y_out;

        // Get initial data for first and last layer
        mpi::scatter(world, v_temp, inp_data, 0);
        if (rank == 1){
            x = inp_data;
        }
        if (rank == N - 1){
            Y_out = inp_data;
        }


        for (int e=0; e < 10; e++){
            // FORWARD
            if (rank != 1){
                world.recv(rank-1, FORWARD_TAG, x);
            }
            y = x+2;
            std::cout << y << "  -rank(forw)= " << rank << std::endl;
            if (rank != N-1){
                world.send(rank+1, FORWARD_TAG, y);
            }

            // BACKWARD
            if (rank != N-1){
                world.recv(rank+1, BACKWARD_TAG, dA);
            }
            if (rank == N-1){
                dA = Y_out - y;
            }
            dA -= 5;
            std::cout << dA << "  -rank(backw)= " << rank << std::endl;
            if (rank != 1){
                world.send(rank-1, BACKWARD_TAG, dA);
            }
        }
    }
}


int main(int argc, char **argv) {

    // CACHING THE INPUT DATA FOR DNN
    std::string cache_fname = "/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/cache/";
    auto op = SGD(0.01);
    auto data = pre_process_data();
    auto X_train = data[0];
    auto X_test = data[1];
    auto Y_train = data[2];
    auto Y_test = data[3];
    auto l1 = FCLayer(X_train.rows(), 10, "linear","he", 1);
    auto l2 = FCLayer(10, 20, "tanh","he",1);

    auto l3 = FCLayer(20, 20, "relu","he",1);
    auto l4 = FCLayer(20, Y_train.rows(), "sigmoid","he",1);
    auto model = ModelCached(op, "BinaryCrossEntropy");
    model.add(l1);
    model.add(l2);
    model.add(l3);
    model.add(l4);

    // LEARNING PHASE
//    model.fit_cached(X_train, Y_train, 10);
//    run_fit(argc, argv, model, cache_fname);

    // EVALUATING PHASE
    model.complile_cache(cache_fname);

    std::cout << "ACCURACY = " << model.evaluate(X_test, Y_test) << std::endl;

}

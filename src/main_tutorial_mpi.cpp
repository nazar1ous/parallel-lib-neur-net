#include <iostream>
#include "layers/config.h"
#include <fstream>

#include "mpi/dnn_model_cached.h"


std::pair<md, md> read_file_data(std::fstream *in_file){
    std::string temp;
    *in_file >> temp;
    auto n_x = std::stoi(temp);
    *in_file >> temp;
    auto m = std::stoi(temp);
    md X(n_x, m);
    md Y(1, m);

    for (int sample = 0; sample < m; sample++){
        *in_file >> temp;
        Y(0, sample) = std::stod(temp);
        for (int x = 0; x < n_x; ++x){
            *in_file >> temp;
            X(x, sample) = std::stod(temp);
        }
    }

    return std::make_pair(X, Y);
}

std::vector<md> pre_process_data(){

    std::fstream in_file("/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/tests/gen_data/cpp_dataset.txt");
    auto data = read_file_data(&in_file);
    md X = data.first;
    md Y = data.second;
    int diff = X.cols() * 0.15;

    md X_train = X.block(0, 0, X.rows(), X.cols() - diff)/255;
    md Y_train = Y.block(0, 0, Y.rows(), Y.cols() - diff);

    md X_test = X.block(0, X.cols() - diff, X.rows(), diff)/255;
    md Y_test = Y.block(0, Y.cols() - diff, Y.rows(), diff);
    return std::vector{X_train, X_test, Y_train, Y_test};
}


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
//    std::string cache_fname = "/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/";
//    auto op = SGD(0.01);
//    auto data = pre_process_data();
//    auto X_train = data[0];
//    auto X_test = data[1];
//    auto Y_train = data[2];
//    auto Y_test = data[3];
//    auto l1 = FCLayer(X_train.rows(), 10, "linear","he", 1);
//    auto l2 = FCLayer(10, 20, "tanh","he",1);
//
//    auto l3 = FCLayer(20, 20, "relu","he",1);
//    auto l4 = FCLayer(20, Y_train.rows(), "sigmoid","he",1);
//    auto model = ModelCached(op, "BinaryCrossEntropy");
//    model.add(l1);
//    model.add(l2);
//    model.add(l3);
//    model.add(l4);
//    model.fit_cached(X_train, Y_train, 10);
//    run_fit(argc, argv, model, cache_fname);



    std::string cache_fname = "/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/";
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
    model.complile_cache(cache_fname);

    std::cout << "ACCURACY = " << model.evaluate(X_test, Y_test) << std::endl;

}

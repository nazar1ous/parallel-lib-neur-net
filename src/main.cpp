#include <iostream>
#include <boost/program_options.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
//#include "layers/structures_mpi.h"
//#include "models/dnn_model.h"
#include "layers/config.h"
#include <fstream>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include "layers/fc_layer_mpi.h"
#include "layers/loss.h"
#include "mpi/serialization_config.h"
#include "mpi/dnn_model_cached.h"

//#include "layers/filter.h"

#define FORWARD_TAG 0
#define BACKWARD_TAG 1
namespace mpi = boost::mpi;

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



void test_fc_layer_mpi(int argc, char *argv[],
                       const ModelCached& model){
    mpi::environment env(argc, argv);
    mpi::communicator world;
    FCLayer layer;
    int epochs = model.epochs;
    int rank = world.rank();
    int N = world.size();
    std::vector<FCLayer> layers(N);
    md inp_data;
    std::vector<md> v_temp(N);
    std::string loss_type;
    std::vector<int> layers_in_sizes(N);
    std::vector<int> layers_out_sizes(N);
    std::vector<std::string> layers_activations(N);
    std::vector<std::string> layers_normalizations(N);
    int in_size;
    int out_size;
    std::string activation_type;
    std::string normalization_type;
    LossWrapper loss = LossWrapper(loss_type);
    FCLayer temp_layer = model.layers[N-1];

//    double learning_rate;


    bool stage_identifier_forward = true;
    if (world.rank() == 0){
        // get data


        // initialize hyper parameters
        // L - number of layers
        int L = 4;
        double learning_rate = 0.01;
        epochs = 1;
        int batch_size = 32;
        std::vector<BasicOptimizer> ops(L);
        for (int l = 0; l < L; ++l){
            ops[l] = SGD(learning_rate);
        }
        std::string loss_type = "BinaryCrossEntropy";

        layers_in_sizes[1] = X_train.rows(); layers_out_sizes[1] = 10;
        layers_in_sizes[2] = 10; layers_out_sizes[2] = 20;
        layers_in_sizes[3] = 20; layers_out_sizes[3] = 20;
        layers_in_sizes[4] = 20; layers_out_sizes[4] = Y_train.rows();
        layers_activations[1] = "linear";
        layers_activations[2] = "tanh";
        layers_activations[3] = "relu";
        layers_activations[4] = "sigmoid";
        layers_normalizations[1] = "he";
        layers_normalizations[2] = "he";
        layers_normalizations[3] = "he";
        layers_normalizations[4] = "he";

        v_temp[1] = X_train; v_temp[N-1] = Y_train;
        mpi::scatter(world, layers_in_sizes, in_size,0);
        mpi::scatter(world, layers_out_sizes, out_size,0);
        mpi::scatter(world, layers_activations, activation_type,0);
        mpi::scatter(world, layers_normalizations, normalization_type,0);
        mpi::broadcast(world, epochs, 0);
        mpi::broadcast(world, learning_rate, 0);

        mpi::broadcast(world, loss_type, 0);
        mpi::scatter(world, v_temp, inp_data, 0);

    }else{

        md AL;
        md X;
        md Y;
        md dA;
        md Y_out;

        // Get initial data for layers
        mpi::scatter(world, layers_in_sizes, in_size,0);
        mpi::scatter(world, layers_out_sizes, out_size,0);
        mpi::scatter(world, layers_activations, activation_type,0);
        mpi::scatter(world, layers_normalizations, normalization_type,0);
        mpi::broadcast(world, epochs, 0);
        mpi::broadcast(world, learning_rate, 0);
        mpi::broadcast(world, loss_type, 0);
        mpi::scatter(world, v_temp, inp_data, 0);
        if (rank == 1){
            X = inp_data;
        }
        if (rank == N - 1){
            loss = LossWrapper(loss_type);
            Y_out = inp_data;
        }
        layer = FCLayer(in_size, out_size, activation_type, SGD(learning_rate), normalization_type);
        layer.initialize_parameters();


        for (int e=0; e < epochs; e++){
            // FORWARD
            if (rank != 1){
                world.recv(rank-1, FORWARD_TAG, X);
            }
            AL = layer.forward(X);

            if (rank != N-1){
                world.send(rank+1, FORWARD_TAG, AL);
            }

            // BACKWARD
            if (rank != N-1){
                world.recv(rank+1, BACKWARD_TAG, dA);
            }
            if (rank == N-1){
                dA = loss.get_loss_backward(AL, Y_out);
            }
            dA = layer.backward(dA);
            layer.update_parameters(1);
            if (rank != 1){
                world.send(rank-1, BACKWARD_TAG, dA);
            }
        }
    }

}





int main(int argc, char **argv) {
    std::string cache_fname = "/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/cache/dnn_cache.txt";
    auto op = SGD(0.05);
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
    model.fit_cached(X_train, Y_train, 10, cache_fname);
//    auto res = model->evaluate(X_test, Y_test);
//    test_algorithm(argc, argv);

    test_fc_layer_mpi(argc, argv, model);

}

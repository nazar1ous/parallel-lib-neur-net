#include <iostream>
//#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h>


#include "layers/config.h"
#include <unordered_map>
#include "layers/loss.h"
#include <fstream>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include "mpi/serialization_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


#define FORWARD_TAG 0
#define BACKWARD_TAG 1
namespace mpi = boost::mpi;


class ModelCached{
public:
    size_t L{};
    std::vector<FCLayer> layers;
    std::string loss_type;
    md X, Y;
    int epochs{};
    BasicOptimizer optimizer;
    Loss* loss{};

    void add(const FCLayer& layer){
        layers.push_back(layer);
    }

    md forward(const md& X_train){
        md Y_(X_train);
        for (int i = 0; i < layers.size(); ++i){
            Y_ = layers[i].forward(Y_);
        }
        return Y_;
    }

    double get_cost(const md& AL, const md& Y_){
        return loss->get_cost(AL, Y_);
    }

    void fit_cached(const md& X_train, const md& Y_train,
                    int epochs=100){
        X = md(X_train);
        Y = md(Y_train);
        this->epochs = epochs;
    }

    void complile_cache(const std::string& cache_dir){
        for (int l = 0; l < layers.size(); ++l){
            std::ifstream ifs(cache_dir+"layer_"+std::to_string(l)+".txt");
            {
                FCLayer lay;
                boost::archive::text_iarchive ia(ifs);
                ia >> lay;
                this->layers[l]._update_from(lay);
            }

        }
    }

    md predict(const md& X){
        return forward(X);
    }

    double evaluate(const md& X_test, const md& Y_test){

        md Y_hat = predict(X_test);

        md diff = Y_test-Y_hat;
        double good = 0;
        for (int i = 0; i < diff.cols(); ++i){
            if (std::abs(diff(0, i)) <= 0.5){
                good++;
            }
        }
        return good/Y_test.cols();
    }

    ModelCached(BasicOptimizer opt, const std::string& loss_type){
        this->optimizer = opt;
        this->loss_type = loss_type;
        this->loss = new LossWrapper(loss_type);
    }

};


void run_fit(int argc, char *argv[],
                       ModelCached model,
                       const std::string& cache_dir){
    mpi::environment env(argc, argv);
    mpi::communicator world;
    int epochs = model.epochs;
    int rank = world.rank();
    int N = world.size();
    FCLayer layer = model.layers[rank];
    layer.initialize_parameters(SGD(0.01));
    LossWrapper loss;
    md AL;
    md X;
    md Y;
    md dA;
    md Y_out;

    if (rank == 0){
        X = model.X;
    }
    if (rank == N-1){
        loss = LossWrapper(model.loss_type);
        Y_out = model.Y;
    }

    for (int e=0; e < epochs; e++){
        // FORWARD
        if (rank != 0){
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


        if (rank != 0){
            world.send(rank-1, BACKWARD_TAG, dA);
        }
    }
    std::ofstream ofs(cache_dir+"layer_"+std::to_string(rank)+".txt");
    {
        boost::archive::text_oarchive oa(ofs);
        oa << layer;
    }

}
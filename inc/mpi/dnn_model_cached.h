#include <iostream>
//#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h>


#include "layers/config.h"
#include <unordered_map>
#include "layers/fc_layer_mpi.h"
#include "layers/loss.h"


class ModelCached{
public:
    size_t L{};
    std::vector<FCLayer> layers;
    std::string loss_type;
    md X, Y;
    int epochs{};
    std::string cache_filename;
    BasicOptimizer optimizer;
    Loss* loss{};

    void add(const FCLayer& layer){
        layers.push_back(layer);
    }

    md forward(const md& X){
        md Y_(X);
        for (int i = 0; i < L; ++i){
            Y_ = layers[i].forward(Y_);
        }
        return Y_;
    }

    double get_cost(const md& AL, const md& Y_){
        return loss->get_cost(AL, Y_);
    }

    void fit_cached(const md& X_train, const md& Y_train,
                    int epochs=100, const std::string& cache_filename){
        X = md(X_train);
        Y = md(Y_train);
        this->epochs = epochs;
        this->cache_filename = cache_filename;
    }

    void complile_cache(){
        // TODO get all layers
    }

    md predict(const md& X){
        return forward(X);
    }

    double evaluate(md& X_test, const md& Y_test){
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
#include <iostream>
#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include "layers/config.h"
#include <unordered_map>
#include "layers/loss.h"

class Model{
private:
    std::vector<OptimizerWrapper*> optimizers;
public:
    size_t L;
    std::vector<FCLayer*> layers;
    std::vector<std::unordered_map<std::string, md>> caches;
    LossWrapper* loss;

    Model(std::vector<FCLayer*>& layers,
          const std::string& loss_type,
          const std::string& optimizer_type,
        const std::unordered_map<std::string, double>& hparams){

        L = layers.size();
        optimizers = std::vector<OptimizerWrapper*>(L,
                                                    new OptimizerWrapper{optimizer_type, hparams});

        this->layers = layers;
        this->caches = std::vector<std::unordered_map<std::string, md>>(L);
        loss = new LossWrapper{loss_type};
    }

    md forward(const md& X){
        md Y(X);
        for (int i = 0; i < L; ++i){
            Y = layers[i]->forward(Y, caches[i]);
        }
        return Y;
    }

    double get_cost(const md& AL, const md& Y){
        return loss->get_cost(AL, Y);
    }

    void backward(const md& AL, const md& Y){
        md dA = loss->get_loss_backward(AL, Y);
        for (int i = L - 1; i >= 0; --i){
            // cache is updated
            dA = layers[i]->backward(dA, caches[i]);
        }
    }

    void update_parameters(){
        for (int i = 0; i < L; ++i){
            layers[i]->update_params(caches[i], *optimizers[i]);
        }
    }

    void fit(const md& X_train, const md& Y_train,
             int max_iter=100){
        for (int i = 0; i < max_iter; ++i){
            auto AL = forward(X_train);
            backward(AL, Y_train);
            update_parameters();
            std::cout << i << "-- " <<  get_cost(AL, Y_train) << std::endl;
        }
    }

    md predict(const md& X){
        return forward(X);
    }
};
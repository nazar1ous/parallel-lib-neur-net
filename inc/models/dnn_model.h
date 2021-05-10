#include <iostream>
#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include "layers/config.h"
#include <unordered_map>

class Model{
public:
    size_t L;
    std::vector<FCLayer*> layers;
    std::vector<std::unordered_map<std::string, md>> caches;
    // TODO probably put the optimizer here
    //  that is common to all layers
    explicit Model(std::vector<FCLayer*>& layers){
        L = layers.size();
        this->layers = layers;
        this->caches = std::vector<std::unordered_map<std::string, md>>(L);
    }

    md forward(const md& X){
        md Y(X);
        for (int i = 0; i < L; ++i){
            Y = layers[i]->forward(Y, caches[i]);
        }
        return Y;
    }

    void backward(const md& AL, const md& Y){
        // TODO add different losses
        md dA = - ((Y.array()/AL.array() - (1 - Y.array())/(1 - AL.array())));
        for (int i = L - 1; i >= 0; --i){
            // cache is updated
            dA = layers[i]->backward(dA, caches[i]);
        }
    }

    void update_parameters(){
        for (int i = 0; i < L; ++i){
            layers[i]->update_params(caches[i]);
        }
    }

    void fit(const md& X_train, const md& Y_train,
             int max_iter=100){
        for (int i = 0; i < max_iter; ++i){
            auto AL = forward(X_train);
            backward(AL, Y_train);
            update_parameters();
        }
    }

    md predict(const md& X){
        return forward(X);
    }
};
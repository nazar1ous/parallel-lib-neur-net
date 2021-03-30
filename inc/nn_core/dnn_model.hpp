//
// Created by nazariikuspys on 3/30/21.
//

#ifndef PARALLEL_NEURAL_NET_LIB_DNN_MODEL_HPP
#define PARALLEL_NEURAL_NET_LIB_DNN_MODEL_HPP
#include <iostream>
#include <vector>
#include "../inc/nn_core/dnn_layer.hpp"

template<typename T>
class DNNModel{
public:
    int layers_num = 0;
    std::vector<DNNLayer<T>*> layers;
    void propagate_forward();
    void propagate_backward();

    DNNModel(int layers_num){
        this->layers_num = layers_num;
        for (int i = 0; i < layers_num - 1; ++i){
            auto l = new DNNLayer<T>(5, "relu");
            l->init_weights(5);
            layers.push_back(l);
        }
        auto l = new DNNLayer<T>(5, "sigmoid");
        l->init_weights(5);
        layers.push_back(l);
    }



};



#endif //PARALLEL_NEURAL_NET_LIB_DNN_MODEL_HPP

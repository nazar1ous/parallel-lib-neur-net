//
// Created by nazariikuspys on 3/30/21.
//
#include "dnn_layer.hpp"

#ifndef PARALLEL_NEURAL_NET_LIB_OPTIMIZERS_HPP
#define PARALLEL_NEURAL_NET_LIB_OPTIMIZERS_HPP

template<typename T>
class Optimizer{
public:
    virtual void update_parameters(std::vector<DNNLayer<T>> layers);
};

template<typename T>
class GDOptimizer: public Optimizer<T>{
public:
    T learning_rate = 0;
    explicit GDOptimizer(T alpha){
        learning_rate = alpha;
    }

    void update_parameters(std::vector<DNNLayer<T>> layers);
};


#endif //PARALLEL_NEURAL_NET_LIB_OPTIMIZERS_HPP

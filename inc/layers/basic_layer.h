
#ifndef PARALLEL_NEURAL_NET_LIB_BASIC_LAYER_H
#define PARALLEL_NEURAL_NET_LIB_BASIC_LAYER_H
#include <map>
#include <iostream>
#include "layers/activations.h"

template<typename T>
class BasicLayer{

public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;

    ActivationWrapper<T> activation;

    volatile void forward_prop(BasicLayer& prev) = 0;
    volatile void backward_prop(BasicLayer& prev) = 0;


};


#endif //PARALLEL_NEURAL_NET_LIB_BASIC_LAYER_H

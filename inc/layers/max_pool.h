#ifndef PARALLEL_NEURAL_NET_LIB_MAX_POOL_H
#define PARALLEL_NEURAL_NET_LIB_MAX_POOL_H

#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "layers/activations.h"
#include "layers/optimizers.h"
#include "layers/filter.h"
#include <random>
#include <ctime>
#include <unordered_map>


class MaxPooling{
public:
    ActivationWrapper* activation;
    int stride;
    int filter_size;
    size_t out_size;
    convL input_data;

    MaxPooling(convL input_data, int filter_size, int stride,
           const std::string& activation_type){
        this->input_data = input_data;
        this->filter_size = filter_size;
        this->stride = stride;
        this->out_size = (input_data(0, 0).rows() - filter_size) / stride + 1;  //TODO: fix square input
        this->activation = new ActivationWrapper{activation_type};
    }

    md single_layer_pool_forward(size_t depth_index) {
        md out_matrix(out_size, out_size);
        for(size_t i=0; i < out_size; i++) {
            for(size_t j=0; j < out_size; j++) {
                out_matrix(i, j) = input_data(depth_index, 0)
                        .block<filter_size, filter_size>(i*stride, j*stride).maxCoeff();
            }
        }
        return out_matrix;
    }

    convL forward(std::unordered_map<std::string, md>& input_cache){
        convL out_data(input_data.rows(), 1);
        for (size_t i=0; i < input_data.rows(); i++) {
            out_data(i, 0) = single_layer_pool_forward(i);
        }
        input_cache["A_prev"] = input_data;
        return out_data;
    }

};

#endif //PARALLEL_NEURAL_NET_LIB_MAX_POOL_H

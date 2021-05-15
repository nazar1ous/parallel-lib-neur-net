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
#include <omp.h>


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
                        .block(i*stride, j*stride, filter_size, filter_size).maxCoeff();
            }
        }
        return out_matrix;
    }

    convL forward(std::unordered_map<std::string, md>& input_cache,
                  std::unordered_map<std::string, int>& size_cache){
        convL out_data(input_data.rows(), 1);
        #pragma omp parallel for
        for (size_t i=0; i < input_data.rows(); i++) {
            out_data(i, 0) = single_layer_pool_forward(i);
        }
        input_cache["A_prev"] = input_data;
        size_cache["stride"] = stride;
        size_cache["filter_size"] = filter_size;
        return out_data;
    }

    md create_max_mask(md X) {
        md mask(X.rows(), X.cols());
        auto max_val = X.maxCoeff();
        for (size_t i=0; i < X.rows(); i++) {
            for (size_t j=0; j < X.cols(); j++) {
                mask(i, j) = (max_val == X(i, j));
            }
        }
        return mask;
    }

    convL backward(convL dA, std::unordered_map<std::string, md>& input_cache,
                                                      std::unordered_map<std::string, int>& size_cache) {
        // Retrieve information from "cache"
        convL a_prev = input_cache.at("A_prev");
        int stride = size_cache.at("stride");
        int filter_size = size_cache.at("filter_size");

        // Retrieve dimensions
        size_t a_size_prev = a_prev(0, 0).rows();
        size_t a_depth_prev = a_prev.rows();
        size_t a_size = dA(0, 0).rows();
        size_t a_depth = dA.rows();

        // Initialize dA_prev, dW(d_filters)
        convL da_prev(a_depth_prev, 1);
        for(size_t h; h < a_depth_prev; h++) {
            da_prev(h, 0) = Eigen::MatrixXd::Zero(a_size_prev, a_size_prev);
        }

        for (size_t i=0; i < a_size; i++) {
            for (size_t j=0; j < a_size; j++) {
                convL a_prev_slice(a_size, 1);
                for (size_t h=0; h < a_size; h++){
                    a_prev_slice(h, 0) = a_prev(h, 0).block(i, j, filter_size, filter_size);
                    auto mask = create_max_mask(a_prev_slice(h, 0));
                    da_prev(h, 0) += mask * dA(h,  0)(i, j);
                }
            }
        }
        return da_prev;
    }

    md flatten(convL out_data) {
        size_t data_size = out_data(0, 0).rows();
        size_t data_depth = out_data.rows();
        md flattened_data(data_size*data_size, data_depth);
        for(size_t h=0; h < data_depth; h++) {
            out_data(h, 0).resize(data_size*data_size, 1);
            flattened_data.col(h) = out_data(h, 0);
        }
        flattened_data.resize(data_size*data_size*data_depth, 1);
        return flattened_data;
    }

    //TODO: update parameters

};

#endif //PARALLEL_NEURAL_NET_LIB_MAX_POOL_H

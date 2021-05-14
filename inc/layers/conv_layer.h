#ifndef PARALLEL_NEURAL_NET_LIB_CONV_LAYER_H
#define PARALLEL_NEURAL_NET_LIB_CONV_LAYER_H

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


class Conv3D{
public:
    ActivationWrapper* activation;
    std::vector<Filter3D> filters;
    convL input_data;

    Conv3D(convL input_data, std::vector<Filter3D>& filters,
            const std::string& activation_type){
        this->input_data = input_data;
        this->filters = filters;
        this->activation = new ActivationWrapper{activation_type};
    }

    md single_filter_forward(Filter3D filter) {
        auto f = filter.f_size;
        size_t out_size = input_data(0, 0).rows() - filter.f_size + 1;  //TODO: fix square input
        md out_matrix(out_size, out_size);
        convL temp_matrix(filter.f_depth, 1);
        for(size_t i=0; i < out_size; i++) {
            for(size_t j=0; j < out_size; j++) {
                for(size_t h=0; h < filter.f_depth; h++) {
                    temp_matrix(h, 0) = input_data(h, 0).block<f, f>(i, j);
                }
                out_matrix(i, j) = filter.one_conv_step(temp_matrix);
            }
        }
        return out_matrix;
    }

    convL forward(std::unordered_map<std::string, md>& input_cache,
               std::unordered_map<std::string, std::vector<Filter3D>>& filter_cache){
        convL out_data(filters.size(), 1);
        for (size_t i=0; i < filters.size(); i++) {
            out_data(i, 0) = single_filter_forward(filters[i]);
        }
        input_cache["A_prev"] = input_data;
        filter_cache["filters"] = filters;
        return out_data;
    }

};

#endif //PARALLEL_NEURAL_NET_LIB_CONV_LAYER_H

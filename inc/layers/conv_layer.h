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
#include <omp.h>


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
                    temp_matrix(h, 0) = input_data(h, 0).block(i, j, f, f);
                }
                out_matrix(i, j) = filter.one_conv_step(temp_matrix);
            }
        }
        return out_matrix;
    }

    convL forward(std::unordered_map<std::string, md>& input_cache,
                  std::unordered_map<std::string, std::vector<Filter3D>>& filter_cache){
        convL out_data(filters.size(), 1);
#pragma omp parallel for
        for (size_t i=0; i < filters.size(); i++) {
            out_data(i, 0) = single_filter_forward(filters[i]);
        }
        input_cache["A_prev"] = input_data;
        filter_cache["filters"] = filters;
        return out_data;
    }

    std::tuple<convL, std::vector<Filter3D>> backward(convL dZ, std::unordered_map<std::string, md>& input_cache,
                                                      std::unordered_map<std::string, std::vector<Filter3D>>& filter_cache) {
        // Retrieve information from "cache"
        convL a_prev = input_cache.at("A_prev");
        std::vector<Filter3D> filter_prev = filter_cache.at("filters");

        // Retrieve dimensions
        size_t a_size_prev = a_prev(0, 0).rows();
        size_t a_depth_prev = a_prev.rows();
        size_t a_size = dZ(0, 0).rows();
        size_t a_depth = dZ.rows();
        size_t filter_size = filter_prev[0].f_size;
        size_t filter_depth = filter_prev[0].f_depth;

        // Initialize dA_prev, dW(d_filters)
        convL da_prev(a_depth_prev, 1);
        for(size_t h; h < a_depth_prev; h++) {
            da_prev(h, 0) = Eigen::MatrixXd::Zero(a_size_prev, a_size_prev);
        }
        std::vector<Filter3D> d_filters;
        for (size_t h=0; h < filter_prev.size(); h++) {
            auto filter = Filter3D(filter_size, a_depth_prev);
            filter.set_params_to_zero();
            d_filters.push_back(filter);
        }

        for (size_t i=0; i < a_size; i++) {
            for (size_t j=0; j < a_size; j++) {
                convL a_slice(a_size, 1);
                for (size_t k=0; k < a_size; k++){
                    a_slice(k, 0) = a_prev(k, 0).block(i, j, filter_size, filter_size);
                }
                for (size_t h=0; h < a_size; h++){
                    da_prev(h, 0) += filters[h].W(h, 0) * dZ(h, 0)(i, j);
                    for (size_t t=0; t < filter_depth; t++) {
                        d_filters[h].W(t, 0) += a_slice(t, 0) * dZ(h, 0)(i, j);
                    }
                    d_filters[h].b += dZ(h, 0)(i, j);
                }
            }
        }
        return std::tuple(da_prev, d_filters);
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

#endif //PARALLEL_NEURAL_NET_LIB_CONV_LAYER_H

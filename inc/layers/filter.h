#ifndef PARALLEL_NEURAL_NET_LIB_FILTER_H
#define PARALLEL_NEURAL_NET_LIB_FILTER_H

#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "layers/activations.h"
#include <random>
#include <ctime>
#include <unordered_map>


class Filter3D{
public:
    size_t f_size;
    size_t f_depth;
    convL W;
    double b;

    Filter3D(size_t f_size, size_t f_depth) {
        this->f_size = f_size;
        this->f_depth = f_depth;
        this->W = convL(f_depth, 1);
        initialize_parameters();
    }

    void initialize_parameters(){

        std::normal_distribution<double> dis(0, 1);
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i=0; i < f_depth; i++) {
            this->W(i, 0) = md(f_size, f_size).unaryExpr([&](double dummy){return dis(gen);}) * 0.5;
        }
        this->b = dis(gen) * 0.5;
    }

    void set_params_to_zero() {
        for (size_t i=0; i < f_depth; i++) {
            this->W(i, 0) = Eigen::MatrixXd::Zero(f_size, f_size);
        }
    }

    double one_conv_step(convL input_data) {
        double cell_res = 0;
        for (size_t i = 0; i < input_data.rows(); i++) {
            cell_res += input_data(i, 0).cwiseProduct(W(i, 0)).sum();
        }
        cell_res += b;
        return cell_res;
    }

};

#endif //PARALLEL_NEURAL_NET_LIB_FILTER_H

#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "layers/activations.h"
#include <random>
#include <ctime>
#include <unordered_map>
#include "layers/optimizers_mpi.h"


class FCLayer{
public:
    ActivationWrapper* activation;
    BasicOptimizer* optimizer{};
    size_t input_size;
    size_t output_size;
    md W, b, A_prev, Z;
    md dW, db;
    double stddev=1;
    std::string initialization = "normal";

    FCLayer(size_t input_size, size_t output_size,
            const std::string& activation_type,
            const std::string& initialization="normal",
            double stddev=1,
            BasicOptimizer* optimizer=nullptr){
        activation = new ActivationWrapper{activation_type};
        this->input_size = input_size;
        this->output_size = output_size;
        this->stddev = stddev;
        this->initialization = initialization;
        this->optimizer = optimizer;
        initialize_parameters();
    }

    md linear_forward(const md& X){
        md temp = W*X;
        temp.colwise() += b.col(0);
        return temp;
    }
    md linear_backward(const md& dZ){
        auto m = dZ.cols();
        dW = (dZ * A_prev.transpose())/m;
        db = dZ.rowwise().sum()/m;
        return W.transpose() * dZ;

    }

    md forward(const md& X){
        A_prev = X;
        Z = linear_forward(X);
        return activation->activate_forward(Z);
    }

    md backward(const md& dA){
        md dZ = activation->activate_backward(dA, Z);
        return linear_backward(dZ);
    }

    void initialize_parameters(){
        if (this->initialization == "he"){
            stddev = sqrt((double)2/input_size);
        }else if (this->initialization == "xavier"){
            stddev = sqrt((double)6/(input_size+output_size));
        }
        std::normal_distribution<double> dis(0, stddev);
        std::random_device rd;
        std::mt19937 gen(rd());
        W = md(output_size, input_size).unaryExpr([&](double dummy){return dis(gen);});
        b = md(output_size, 1).unaryExpr([&](double dummy){return dis(gen);});
        this->optimizer->init_(get_params(), get_grads());
    }

    std::vector<md*> get_params(){
        return std::vector<md*>{&W, &b};
    }

    void update_parameters(int mini_batch_n){
        this->optimizer->update_parameters(mini_batch_n);
    }


    std::vector<md*> get_grads(){
        return std::vector<md*>{&dW, &db};
    }


};
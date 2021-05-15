#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "layers/activations.h"
#include <random>
#include <ctime>
#include <unordered_map>


class FCLayer{
public:
    ActivationWrapper* activation;
    size_t input_size;
    size_t output_size;
    md W, b;
    double stddev=1;
    std::string initialization = "normal";

    FCLayer(size_t input_size, size_t output_size,
            const std::string& activation_type,
            const std::string& initialization="normal",
            double stddev=1){
        activation = new ActivationWrapper{activation_type};
        this->input_size = input_size;
        this->output_size = output_size;
        this->stddev = stddev;
        this->initialization = initialization;
        initialize_parameters();
    }

    md linear_forward(const md& X){
        md temp = W*X;
        temp.colwise() += b.col(0);
        return temp;
    }
    md linear_backward(const md& dZ, std::unordered_map<std::string, md>& cache){
        auto m = dZ.cols();
        cache["dW"] = (dZ * cache["A_prev"].transpose())/m;
        cache["db"] = dZ.rowwise().sum()/m;
        return W.transpose() * dZ;

    }

    md forward(const md& X, std::unordered_map<std::string, md>& cache){
        cache["A_prev"] = X;
        md Z = linear_forward(X);
        cache["Z"] = Z;
        return activation->activate_forward(Z);
    }

    md backward(const md& dA, std::unordered_map<std::string, md>& cache){
        md dZ = activation->activate_backward(dA, cache["Z"]);
        return linear_backward(dZ, cache);
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
    }

    std::vector<md*> get_params(){
        return std::vector<md*>{&W, &b};
    }


    static std::vector<std::string> get_grads(){
        return std::vector<std::string>{"dW", "db"};
    }


};
#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "layers/activations.h"
#include "layers/optimizers.h"
#include <random>
#include <ctime>
#include <unordered_map>


//md HStack(const md& d, int m) {
//    int rows_n = d.rows();
//    md vstacked_mat(rows_n, m);
//    int col_offset = 0;
//    for (int i = 0; i < m; ++i) {
//        vstacked_mat.middleCols(col_offset, 1) = d;
//        col_offset +=  1;
//    }
//    return vstacked_mat;
//}



class FCLayer{
public:
    OptimizerWrapper* optimizer;
    ActivationWrapper* activation;
    size_t input_size;
    size_t output_size;
    md W, b;

    FCLayer(size_t input_size, size_t output_size,
            const std::string& optimizer_type,
            const std::string& activation_type,
            const std::unordered_map<std::string, double>& hparams){
        optimizer = new OptimizerWrapper{optimizer_type, hparams};
        activation = new ActivationWrapper{activation_type};
        this->input_size = input_size;
        this->output_size = output_size;
        initialize_parameters();
    }

    md linear_forward(const md& X){
        md temp = W*X;
        temp.colwise() += b.col(0);
        return temp;
    }
    md linear_backward(const md& dZ, std::unordered_map<std::string, md>& cache){
        auto m = dZ.cols();
        cache["dW"] = (1/m) * (dZ * cache["A_prev"].transpose());
        cache["db"] = (1/m) * dZ.rowwise().sum();
        return W.transpose() * dZ;

    }

    md forward(const md& X, std::unordered_map<std::string, md>& cache){
        cache["A_prev"] = X;
        auto Z = linear_forward(X);
        cache["Z"] = Z;
        return activation->activate_forward(Z);
    }

    md backward(const md& dA, std::unordered_map<std::string, md>& cache){
        auto dZ = activation->activate_backward(dA, cache["Z"]);
        return linear_backward(dZ, cache);
    }

    void update_params(std::unordered_map<std::string, md>& cache){
        optimizer->update_parameters(&W, &b, cache);
    }

    void initialize_parameters(){
        std::normal_distribution<double> dis(0, 1);
        std::random_device rd;
        std::mt19937 gen(rd());
        W = md(output_size, input_size).unaryExpr([&](double dummy){return dis(gen);});
        b = md(output_size, 1).unaryExpr([&](double dummy){return dis(gen);});
    }


};
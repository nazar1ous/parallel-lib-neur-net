#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
//#include "fc_layer.h"
#include "layers/activations.h"
#include "layers/optimizers.h"


template<typename T>
class FCLayer{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    OptimizerWrapper<T>* optimizer;
    ActivationWrapper<T>* activation;
    size_t input_size;
    size_t output_size;
    MatrixTx W, b;

    FCLayer(size_t input_size, size_t output_size,
            const std::string& optimizer_type,
            const std::string& activation_type,
            const std::unordered_map<std::string, T>& hparams){
        optimizer = new OptimizerWrapper<T>{optimizer_type, hparams};
        activation = new ActivationWrapper<T>{activation_type};
        this->input_size = input_size;
        this->output_size = output_size;
        W.resize(output_size, input_size);
        b.resize(output_size, 1);
    }

    MatrixTx linear_forward(const MatrixTx& X){
        return W*X + b;
    }
    MatrixTx linear_backward(const MatrixTx& dZ, std::unordered_map<std::string, MatrixTx>& cache){
        auto m = dZ.cols();
        cache["dW"] = (T(1)/m) * (dZ * cache["A_prev"].transpose());
        cache["db"] = (T(1)/m) * dZ.rowwise().sum();
        return W.transpose() * dZ;

    }

    MatrixTx forward(const MatrixTx& X, std::unordered_map<std::string, MatrixTx>& cache){
        cache["A_prev"] = X;
        auto Z = linear_forward(X);
        cache["Z"] = Z;
        return activation->activate_forward(Z);
    }

    MatrixTx backward(const MatrixTx& dA, std::unordered_map<std::string, MatrixTx>& cache){
        auto dZ = activation->activate_backward(dA, cache["Z"]);
        return linear_backward(dZ, cache);
    }

    void update_params(const std::unordered_map<std::string, MatrixTx>& cache){
        optimizer->update_parameters(&W, &b, cache);
    }


};
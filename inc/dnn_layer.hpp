#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <map>
using Eigen::MatrixXd;


template<typename T>
class DNNLayer{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string activation_f;
    MatrixTx A, Z, dW, db;
    MatrixTx W;
    MatrixTx b;
    std::vector<std::string> SUPPORTED_FUNCTIONS = {
            "relu", "sigmoid", "tanh"
    };
    size_t neurons_num = 0;

    DNNLayer(size_t cur_layer_neuron_num, std::string activation_f){

        this->neurons_num = cur_layer_neuron_num;
        this->activation_f = activation_f;
        if (!find_func(activation_f) || cur_layer_neuron_num == 0){
            exit(1);
        }
    }


    bool find_func(std::string func_name) const{
        for (auto name: this->SUPPORTED_FUNCTIONS){
            if (name == func_name)
                return true;
        }
        return false;
    }

//    DNNLayer(size_t cur_layer_neuron_num, std::string activation_f);

    void init_weights(size_t prev_layer_neuron_num);


    static T sigmoid_der(T x){
        return sigmoid(x) * (1-sigmoid(x));
    }
    static T sigmoid(T x){
        return 1/(1 + exp(-x));
    }

    static T relu(T x){
        return max(0, x);
    }

    static T relu_der(T x){
        if (x < 0){
            return 0;
        }
        // Undefined in x= 0
        return 1;
    }

    static T tan_h(T x){
        return tanh(x);
    }

    static T tan_h_der(T x){
        return 1 - pow(tan_h(x), 2);
    }

    T ac_func(T x);
    T ac_func_der(T x);

};

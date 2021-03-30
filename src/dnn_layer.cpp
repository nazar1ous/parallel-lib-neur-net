#include "dnn_layer.hpp"


# define tt template<typename T>
# define lr DNNLayer<T>

tt
void lr::init_weights(size_t prev_layer_neuron_num) {
    if (this->neurons_num == 0)
        exit(1);
    if (this->activation_f == "relu"){
        W = (MatrixTx::Random(neurons_num, prev_layer_neuron_num)
             *2/std::sqrt(this->neurons_num));
    }else{
        W = (MatrixTx::Random(neurons_num, prev_layer_neuron_num)
             /std::sqrt(this->neurons_num));
    }
    b = MatrixTx::Zero(neurons_num, 1);
}

tt
T lr::ac_func (T x){
    if (this->activation_f == "sigmoid"){
        return this->sigmoid(x);
    }
    else if (this->activation_f == "relu"){
        return this->relu(x);
    } else if (this->activation_f == "tanh"){
        return this->tan_h(x);
    }
}

tt
T lr::ac_func_der(T x){
    if (this->activation_f == "sigmoid"){
        return this->sigmoid_der(x);
    }
    else if (this->activation_f == "relu"){
        return this->relu_der(x);
    } else if (this->activation_f == "tanh"){
        return this->tan_h_der(x);
    }
}

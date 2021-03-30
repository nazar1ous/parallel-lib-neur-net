//
// Created by nazariikuspys on 3/30/21.
//
#include <optimizers.hpp>

template<typename T>
void Optimizer<T>::update_parameters(std::vector<DNNLayer<T>> layers) {

}

template<typename T>
void GDOptimizer<T>::update_parameters(std::vector<DNNLayer<T>> layers) {

    for (const auto &l: layers){
        *(l.W) -= learning_rate* *(l.dW);
        *(l.b) -= learning_rate* *(l.dW);
    }
}

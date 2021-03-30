
#include "../inc/nn_core/dnn_model.hpp"
template<typename T>
void DNNModel<T>::propagate_forward() {
    for (const auto &l: layers){
        std::cout << *l.dW << std::endl;
    }
}

template<typename T>
void DNNModel<T>::propagate_backward() {

}
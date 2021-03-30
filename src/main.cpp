#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
#include <bits/stdc++.h>
#include <Eigen/Dense>
//#include "../inc/nn_core/dnn_model.hpp"
#include "../inc/nn_core/dnn_layer.hpp"
//#include "../src/nn_core/dnn_layer.cpp"






int main(int argc, char **argv) {
    auto l = new DNNLayer<double>(5, "sigmoid");
    l->init_weights(5);
    std::cout << l->W << std:: endl;
//    l->dW *= 5;
//    auto model = DNNModel<double>(2);

}

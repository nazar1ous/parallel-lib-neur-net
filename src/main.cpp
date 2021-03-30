#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "dnn_layer.hpp"






int main(int argc, char **argv) {
//    DNNLayer<double>(5, "sigmoid");
    auto l = new DNNLayer<double>(5, "sigmoid");
    l->init_weights(5);
//    std::cout << l->W << std:: endl;
//    l->dW *= 5;
//    auto model = DNNModel<double>(2);

}

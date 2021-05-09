#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
//#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "models/dnn_model.h"

void test_fc_layer_basic(){
    std::unordered_map<std::string, double> m;
    m["alpha"] = 0.1;
    auto l = new FCLayer<double>(5, 5, "gd", "sigmoid", m);
    std::cout << l->input_size;
    std::cout << l->optimizer->hparams["alpha"];
    std::cout << l->activation->type << std::endl;
    std::cout << l->W << std::endl;
    std::cout << l->b << std::endl;
}





int main(int argc, char **argv) {

    test_fc_layer_basic();

}

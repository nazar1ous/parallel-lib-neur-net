#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
//#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "models/dnn_model.h"





int main(int argc, char **argv) {
    std::unordered_map<std::string, double> m;
    m["alpha"] = 0.1;
    auto l = new FCLayer<double>(5, 5, "gd", "sigmoid", m);
    std::cout << l->input_size;
    std::cout << l->optimizer->hparams["alpha"];
    std::cout << l->activation->type;

}

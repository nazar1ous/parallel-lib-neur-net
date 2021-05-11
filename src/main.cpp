#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
//#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "models/dnn_model.h"
#include "layers/config.h"
#include <fstream>

void read_file_to_vector(std::vector<double> *v, std::fstream *in_file){
    std::string temp;
    while (*in_file >> temp){
        v->push_back(std::stod(temp));
    }
}

void test_fc_layer_basic(){

    std::unordered_map<std::string, double> m;
    m["alpha"] = 0.01;
    auto l1 = new FCLayer(1, 2, "gd", "relu", m);
    auto l2 = new FCLayer(2, 1, "gd", "relu", m);
//    auto l3 = new FCLayer(10, 1, "gd", "sigmoid", m);
    std::vector<FCLayer*> layers = {
            l1,
            l2
    };
    auto model = new Model(layers, "mse");
    std::fstream in_file("/home/nazariikuspys/temp/data.txt");
    std::vector<double> v;
    read_file_to_vector(&v, &in_file);
    int n = (int)v[0];
    std::cout << n;
    md X_train = md (1, n*2);
    md Y_train = md (1, n*2);
    int j = 0;
    for (int i = 1; i < n*4 - 1; i+=2){
        X_train(0, j) = v[i];
        Y_train(0, j) = v[i+1];
        j++;
    }

    model->fit(X_train, Y_train);

//    std::cout << std::endl;
//    std::cout << X_train << std::endl;
//    std::cout << Y_train << std::endl;

//    std::cout << l->input_size;
//    std::cout << l->optimizer->hparams["alpha"];
//    std::cout << l->activation->type << std::endl;
//    std::cout << l->W << std::endl;
//    std::cout << l->b << std::endl;
//    std::unordered_map<std::string, md> kal;
//    kal.insert(std::make_pair("A_prev", X_train));
}





int main(int argc, char **argv) {

    test_fc_layer_basic();

}

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
    auto l1 = new FCLayer(1, 3, "sigmoid");
    auto l2 = new FCLayer(3, 5, "linear");
    auto l3 = new FCLayer(5, 10, "tanh");
    auto l4 = new FCLayer(10, 1, "relu");


    std::vector<FCLayer*> layers = {
            l1,
            l2,
            l3,
            l4
    };
    auto model = new Model(layers);
//    auto op = new SGD(0.01);
//    auto op = new RMSprop(0.01);
//    auto op = new RMSprop(0.01);
    auto op = new GDWithMomentum(0.01);

    model->compile("mse", op);
    std::fstream in_file("/home/nazariikuspys/temp/data.txt");
    std::vector<double> v;
    read_file_to_vector(&v, &in_file);
    int n = (int)v[0];
    md X_train = md (1, n*2);
    md Y_train = md (1, n*2);
    int j = 0;
    for (int i = 1; i < n*4 - 1; i+=2){
        X_train(0, j) = v[i];
        Y_train(0, j) = v[i+1];
        j++;
    }

    model->fit(X_train, Y_train, 50, true);
}





int main(int argc, char **argv) {

    test_fc_layer_basic();

}

#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
//#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "models/dnn_model.h"
#include "layers/config.h"
#include <fstream>


std::pair<md, md> read_file_data(std::fstream *in_file){
    std::string temp;
    *in_file >> temp;
    auto n_x = std::stoi(temp);
    *in_file >> temp;
    auto m = std::stoi(temp);
    md X(n_x, m);
    md Y(1, m);

    for (int sample = 0; sample < m; sample++){
        *in_file >> temp;
        Y(0, sample) = std::stod(temp);
        for (int x = 0; x < n_x; ++x){
            *in_file >> temp;
            X(x, sample) = std::stod(temp);
        }
    }

    return std::make_pair(X, Y);
}

void test_fc_layer_basic(){

    std::fstream in_file("./tests/gen_data/cpp_dataset.txt");
    auto data = read_file_data(&in_file);
    md X = data.first;
    md Y = data.second;
    int diff = X.cols() * 0.15;

    md X_train = X.block(0, 0, X.rows(), X.cols() - diff)/255;
    md Y_train = Y.block(0, 0, Y.rows(), Y.cols() - diff);

    md X_test = X.block(0, X.cols() - diff, X.rows(), diff)/255;
    md Y_test = Y.block(0, Y.cols() - diff, Y.rows(), diff);

    //    auto op = new SGD(0.01);
//    model->fit_data_parallel(X_train, Y_train, 10, true);



    auto l1 = new FCLayer(X.rows(), 10, "linear", "he");
    auto l2 = new FCLayer(10, 20, "tanh", "he");

    auto l3 = new FCLayer(20, 20, "relu", "he");
    auto l4 = new FCLayer(20, Y.rows(), "sigmoid", "he");
    auto model = new Model{};
    model->add(l1);
    model->add(l2);
    model->add(l3);
    model->add(l4);
    auto op = new Adam(0.01);
    auto loss = new BinaryCrossEntropy();
    model->compile(loss, op);
    model->fit(X_train, Y_train, 50, true, 24);
    auto res = model->evaluate(X_test, Y_test);
    std::cout << "Test accuracy = " << std::to_string(res) << std::endl;

}





int main(int argc, char **argv) {

    test_fc_layer_basic();

}

#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
//#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "models/dnn_model.h"


template<typename T>
void test_fc_layer_basic(){
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;

    std::unordered_map<std::string, double> m;
    m["alpha"] = 0.01;
    auto l1 = new FCLayer<double>(1, 20, "gd", "relu", m);
    auto l2 = new FCLayer<double>(20, 10, "gd", "relu", m);
    auto l3 = new FCLayer<double>(10, 1, "gd", "sigmoid", m);
    std::vector<FCLayer<double>*> layers = {
            l1,
            l2,
            l3
    };
    auto model = new Model<double>(layers);
    int n;
    std::cin >> n;
    double l, k;
    auto X_train = MatrixTx (1, n);
    auto Y_train = MatrixTx (1, n);

    for (int i = 0; i < n; ++i){
        std::cin >> l >> k;
        X_train(0, i) = l;
        Y_train(0, i) = k;

    }
    std::cout << X_train.cols() << X_train.rows();
    sleep(1);
    model->fit(X_train, Y_train);
//    std::cout << std::endl;
//    std::cout << X_train << std::endl;
//    std::cout << Y_train << std::endl;

//    std::cout << l->input_size;
//    std::cout << l->optimizer->hparams["alpha"];
//    std::cout << l->activation->type << std::endl;
//    std::cout << l->W << std::endl;
//    std::cout << l->b << std::endl;
}





int main(int argc, char **argv) {

    test_fc_layer_basic<double>();

}

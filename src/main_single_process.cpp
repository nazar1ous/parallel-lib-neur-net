#include <iostream>
#include <fstream>
#include "models/dnn_model.h"
#include "process_data/process_data.h"


int main(int argc, char **argv) {
    auto data = pre_process_data();
    auto X_train = data[0];
    auto X_test = data[1];
    auto Y_train = data[2];
    auto Y_test = data[3];

    auto l1 = new FCLayer(X_train.rows(), 10, "linear", "he", 1);
    auto l2 = new FCLayer(10, 20, "tanh", "he",1);

    auto l3 = new FCLayer(20, 20, "relu", "he",1);
    auto l4 = new FCLayer(20, Y_train.rows(), "sigmoid", "he",1);
    auto model = new Model{};
    model->add(l1);
    model->add(l2);
    model->add(l3);
    model->add(l4);
    auto op = new SGD(0.01);
    auto loss = new BinaryCrossEntropy();
    model->compile(loss, op);
    model->fit(X_train, Y_train, 10, true, Y_train.cols());
    auto res = model->evaluate(X_test, Y_test);
    std::cout << "Test accuracy = " << std::to_string(res) << std::endl;

}
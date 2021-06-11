#include <iostream>
#include <boost/program_options.hpp>
//#include "models/dnn_model.h"
//#include "layers/config.h"
#include <fstream>
//#include "layers/filter.h"
//#include "models/dnn_model_mpi.h"
#include "models/dnn_model.h"
#include "process_data/process_data.h"


void test_fc_layer_basic(){
//    std::fstream in_file("/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/tests/dataset.txt");

//    auto op = new SGD(0.01);
//    md X_train(1, 1);
//    md Y_train(1, 1);
//    X_train(0, 0) = 1;
//    Y_train(0, 0) = 1;
    auto data = pre_process_data();
    auto X_train = data[0];
    auto X_test = data[1];
    auto Y_train = data[2];
    auto Y_test = data[3];

    auto l1 = new FCLayer(X_train.rows(), 100, "linear", "he", 1);
    auto l2 = new FCLayer(100, 200, "tanh", "he",1);

    auto l3 = new FCLayer(200, 20, "relu", "he",1);
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





int main(int argc, char **argv) {

    test_fc_layer_basic();

}

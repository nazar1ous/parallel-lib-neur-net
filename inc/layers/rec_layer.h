#ifndef PARALLEL_NEURAL_NET_LIB_REC_LAYER_H
#define PARALLEL_NEURAL_NET_LIB_REC_LAYER_H

#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "layers/activations.h"
#include <random>
#include <ctime>
#include <unordered_map>


class RLayer {
public:
    ActivationWrapper* activation_a;
    ActivationWrapper* activation_x;
    std::string initialization;

    int n_a;
    int n_x;
    int n_y;
    double stddev_a;
    double stddev_x;

    std::unordered_map<std::string, md> parameters;

    RLayer(int n_a, int n_x, int n_y,
    const std::string& activation_type_a,
    const std::string& activation_type_x,
    const std::string& initialization="normal",
    double stddev_a=1, double stddev_x=1){
        activation_a = new ActivationWrapper{activation_type_a};
        activation_x = new ActivationWrapper{activation_type_x};
        this->n_a = n_a;
        this->n_x = n_x;
        this->n_y = n_y;
        this->stddev_a = stddev_a;
        this->stddev_x = stddev_x;
        this->initialization = initialization;
        initialize_parameters(n_a, n_x, n_y);

    }

    void initialize_parameters(int num_a, int num_x, int num_y){
        if (this->initialization == "he"){
            stddev_a = sqrt((double)2/num_a);
            stddev_x = sqrt((double)2/num_x);
        }else if (this->initialization == "xavier"){
            stddev_a = sqrt((double)6/(num_a + num_a));
            stddev_x = sqrt((double)6/(num_a + num_x));
        }
        std::normal_distribution<double> dis_a(0, stddev_a);
        std::normal_distribution<double> dis_x(0, stddev_x);
        std::random_device rd;
        std::mt19937 gen(rd());
        md Wax = md(num_a, num_x).unaryExpr([&](double dummy){return dis_x(gen);})*0.01;
        md Waa = md(num_a, num_a).unaryExpr([&](double dummy){return dis_a(gen);})*0.01;
        md Wya = md(num_y, num_a).unaryExpr([&](double dummy){return dis_x(gen);})*0.01;

        md b = md::Zero(num_a, 1);
        md by = md::Zero(num_y, 1);

        parameters.insert({{"Waa", Waa}, {"Wax", Wax}, {"Wya", Wya}, {"b", b}, {"by", by}});
    }

    void update_parameters(std::unordered_map<std::string, md> gradients,
                                                          double learning_rate) {
        parameters["Waa"] -= learning_rate * gradients["dWaa"];
        parameters["Wax"] -= learning_rate * gradients["dWax"];
        parameters["Wya"] -= learning_rate * gradients["dWya"];
        parameters["b"] -= learning_rate * gradients["db"];
        parameters["by"] -= learning_rate * gradients["dby"];
    }

    std::pair<md, md> rnn_step_forward(md a_prev, md x) {
        auto Waa = parameters["Waa"];
        auto Wax = parameters["Wax"];
        auto Wya = parameters["Wya"];
        auto b = parameters["b"];
        auto by = parameters["by"];

        md vb = md::Zero(b.rows(), a_prev.cols());
        md vby = md::Zero(by.rows(), a_prev.cols());

        for (int i=0; i < a_prev.cols(); i++) {
            vb.col(i) += b.col(0);
            vby.col(i) += by.col(0);
        }

        md a_next = activation_a->activate_forward(Wax * x + Waa * a_prev + vb);
        md y_pred = activation_x->activate_forward(Wya * a_next + vby);

        return std::pair(a_next, y_pred);
    }

    std::unordered_map<std::string, md> rnn_step_backward(md dy, std::unordered_map<std::string, md> *gradients,
                                                          md x, md a, md a_prev) {

        (*gradients)["dWya"] += dy * a.transpose();
        (*gradients)["dby"] += dy;
        md da = parameters["Wya"].transpose() * dy + (*gradients)["da_next"];
        md da_raw = da.array() * (1 - a.array().square());
        (*gradients)["db"] += da_raw;
        (*gradients)["dWax"] += da_raw * x.transpose();
        (*gradients)["dWaa"] += da_raw * a_prev.transpose();
        (*gradients)["da_next"] += parameters["Waa"].transpose() * da_raw;
        return *gradients;
    }

    std::pair<double, std::unordered_map<std::string, std::unordered_map<int, md>>> rnn_forward(std::vector<int> X, std::vector<int> Y,
                                                                                                md a0, int vocabulary_size=27) {
        std::unordered_map<std::string, std::unordered_map<int, md>> cache;
        std::unordered_map<int, md> x, a, y_hat;
        md a_1 = a0.col(0);
        a.insert({{-1, a_1}});
        double loss = 0;

        for (int t = 0; t < X.size(); t++) {
            x[t] = md::Zero(vocabulary_size, 1);
            if (X[t] != -1000) {
                x[t](X[t], 0) = 1;
            }

            auto f_pair = rnn_step_forward(a[t-1],x[t]);
            a[t] = f_pair.first;
            y_hat[t] = f_pair.second;
            loss -= log(y_hat[t](Y[t], 0));
        }

        cache.insert({{"y_hat", y_hat}, {"a", a}, {"x", x}});
        return std::pair(loss, cache);
    }

    std::pair<std::unordered_map<std::string, md>, std::unordered_map<int, md>> rnn_backward(std::vector<int> X, std::vector<int> Y,
                                                                                             std::unordered_map<std::string, std::unordered_map<int, md>> cache) {
        std::unordered_map<std::string, md> gradients;

        auto y_hat = cache["y_hat"];
        auto a = cache["a"];
        auto x = cache["x"];

        auto Waa = parameters["Waa"];
        auto Wax = parameters["Wax"];
        auto Wya = parameters["Wya"];
        auto b = parameters["b"];
        auto by = parameters["by"];

        md dWax = md::Zero(Wax.rows(), Wax.cols());
        md dWaa = md::Zero(Waa.rows(), Waa.cols());
        md dWya = md::Zero(Wya.rows(), Wya.cols());
        md db = md::Zero(b.rows(), b.cols());
        md dby = md::Zero(by.rows(), by.cols());
        md da_next = md::Zero(a[0].rows(), a[0].cols());

        gradients.insert({{"dWaa", dWaa}, {"dWax", dWax}, {"dWya", dWya},
                          {"db", db}, {"dby", dby}, {"da_next", da_next}});

        for (int t = X.size()-1; t > -1; t--) {
            md dy = y_hat[t];
            dy(Y[t], 0) -= 1;
            gradients = rnn_step_backward(dy, &gradients,
                                          x[t], a[t], a[t-1]);
        }
        return std::pair(gradients, a);
    }

    std::unordered_map<std::string, md> clip(std::unordered_map<std::string, md> gradients, double max_value) {
        std::unordered_map<std::string, md> new_gradient;
        auto dWaa = gradients["dWaa"];
        auto dWax = gradients["dWax"];
        auto dWya = gradients["dWya"];
        auto db = gradients["db"];
        auto dby = gradients["dby"];
        std::vector<md> a = {dWaa, dWax, dWya, db, dby};

        for (int k=0; k < a.size(); k++) {
            for (int i=0; i < a[k].rows(); i++) {
                for (int j=0; j < a[k].cols(); j++){
                    if (a[k](i, j) > max_value) {
                        a[k](i, j) = max_value;
                    }
                    if (a[k](i, j) < -max_value) {
                        a[k](i, j) = -max_value;
                    }
                }
            }
        }
        new_gradient.insert({{"dWaa", a[0]}, {"dWax", a[1]}, {"dWya", a[2]},
                             {"db", a[3]}, {"dby", a[4]}});
        return new_gradient;
    }

};


#endif //PARALLEL_NEURAL_NET_LIB_REC_LAYER_H

#ifndef PARALLEL_NEURAL_NET_LIB_RNN_MODEL_H
#define PARALLEL_NEURAL_NET_LIB_RNN_MODEL_H

#include <iostream>
#include "layers/rec_layer.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h>

#include "layers/config.h"
#include <unordered_map>
#include <fstream>


class RNNModel {
public:

    RLayer* layer;
    std::unordered_map<char, int> char_to_ix;
    std::unordered_map<int, char> ix_to_char;

    explicit RNNModel(RLayer *layer, std::unordered_map<char, int> char_to_ix,
                      std::unordered_map<int, char> ix_to_char) {
        this->layer = layer;
        this->char_to_ix = char_to_ix;
        this->ix_to_char = ix_to_char;
    }

    double get_initial_loss(int vocabulary_size, int seq_length){
        return -log(1.0/vocabulary_size)*seq_length;
    }

    double smooth(double loss, double cur_loss) {
        return loss * 0.999 + cur_loss * 0.001;
    }

    std::vector<std::string> read_data(std::string data_file) {
        std::ifstream file(data_file);
        std::string str;
        std::vector<std::string> str_vec;
        while (std::getline(file, str)) {
            str_vec.push_back(boost::to_lower_copy(boost::trim_copy(str)));
        }
        file.close();
        return str_vec;
    }




    std::tuple<double, std::unordered_map<std::string, md>, md> optimize(std::vector<int> X, std::vector<int> Y,
                                                                         md a_prev, double learning_rate=0.01) {
        std::pair<double, std::unordered_map<std::string, std::unordered_map<int, md>>> forward_pair;
        forward_pair = this->layer->rnn_forward(X, Y, a_prev);
        auto loss = forward_pair.first;
        auto cache = forward_pair.second;

        std::pair<std::unordered_map<std::string, md>, std::unordered_map<int, md>> backward_pair;
        backward_pair = this->layer->rnn_backward(X, Y, cache);
        auto gradients = backward_pair.first;
        auto a = backward_pair.second;

        auto clipped_gradients = layer->clip(gradients, 5);

        this->layer->update_parameters(clipped_gradients, learning_rate);

        return std::tuple(loss, gradients, a[X.size()-1]);
    }


    std::vector<int> sample(std::unordered_map<std::string, md> parameters) {
        auto Waa = parameters["Waa"];
        auto Wax = parameters["Wax"];
        auto Wya = parameters["Wya"];
        auto b = parameters["b"];
        auto by = parameters["by"];
        int vocab_size = by.rows();
        int n_a = Waa.cols();

        md x = md::Zero(vocab_size, 1);
        md a_prev = md::Zero(n_a, 1);

        std::vector<int> indices;
        int idx = -1;
        char newline_char = char_to_ix['\n'];

        md a, z, y;

        while (idx != newline_char) {
            md vb(b.rows(), a_prev.cols());
            md vby(by.rows(), a_prev.cols());

            for (int i=0; i < a_prev.cols(); i++) {
                vb.col(i) = b.col(0);
                vby.col(i) = by.col(0);
            }

            a = layer->activation_a->activate_forward(Wax * x + Waa * a_prev + vb);
            z = Wya * a + vby;

            md y = layer->activation_x->activate_forward(z);

            std::vector<double> distr;
            std::random_device rd;
            std::mt19937 gen(rd());
            Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd>(y.data(), y.size()).transpose();
            for (int ind = 0; ind < v.size(); ind++) {
                distr.push_back(v[ind]);
            }
            std::discrete_distribution<> d(distr.begin(), distr.end());
            idx = d(gen);

            indices.push_back(idx);
            x = md::Zero(vocab_size, 1);
            x(idx, 0) = 1;
            a_prev = a;
        }
        return indices;
    }

    void print_sample(std::vector<int> sample_ix) {
        std::string txt = "";
        for (auto &ix: sample_ix) {
            txt += ix_to_char[ix];
        }
        txt[0] = std::toupper(txt[0]);
        std::cout << txt;
    }

    void generate(std::unordered_map<std::string, md> parameters_gen) {
        std::vector<int> sample_ix;
        sample_ix = sample(parameters_gen);
        print_sample(sample_ix);
    }

    std::unordered_map<std::string, md> train(std::string data_file, Loss* loss_r, int num_iterations=35000,
                                              bool generate=false, int n_a=50, int num_names=7,
                                              int vocabulary_size=27){
        auto loss = get_initial_loss(vocabulary_size, num_names);

        std::vector<std::string> examples = read_data(data_file);
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(examples), std::end(examples), rng);

        md a_prev = md::Zero(n_a, 1);
        std::vector<int> sampled_indices;

        for (int j = 0; j < num_iterations; j++) {
            std::vector<int> X, Y;
            X.push_back(-1000);
            int index = j % examples.size();
            for (int idx_1 = 0; idx_1 < std::size(examples[index]); idx_1++) {
                X.push_back(char_to_ix[examples[index][idx_1]]);
            }
            Y = std::vector<int>(X.begin() + 1, X.end());
            Y.push_back(char_to_ix['\n']);

            std::tuple<double, std::unordered_map<std::string, md>, md> opt_tuple =
                    optimize(X, Y, a_prev);
            double cur_loss = std::get<0>(opt_tuple);
            std::unordered_map<std::string, md> gradients = std::get<1>(opt_tuple);
            a_prev = std::get<2>(opt_tuple);

            loss = smooth(loss, cur_loss);

            if (j % 2000 == 0) {
                std::cout << "Iteration: " << j << ", Loss = " << loss << std::endl;
                if (generate) {
                    for (int k = 0; k < num_names; k++) {
                        sampled_indices = sample(this->layer->parameters);
                        print_sample(sampled_indices);
                    }
                }
            }
        }
        return this->layer->parameters;
    }
};

#endif //PARALLEL_NEURAL_NET_LIB_RNN_MODEL_H
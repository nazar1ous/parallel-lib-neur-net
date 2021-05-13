#include <iostream>
//#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include <Eigen/Core>

#include "layers/config.h"
#include <unordered_map>
#include "layers/loss.h"
#include "layers/optimizers.h"

class Model{
private:
    BasicOptimizer* optimizer;

    static std::vector<std::pair<md, md>> split_data(const md& X_train,
                                                     const md& Y_train,
                                                     int batch_size){
        if (batch_size > Y_train.cols()){
            std::cerr << "Not valid batch size" << std::endl;
            exit(1);
        }
        std::vector<std::pair<md, md>> split_data_;
        int k = batch_size;
        int curI = 0;
        for (int i = 0; i <= Y_train.cols()-k+1; i+=k){
            split_data_.emplace_back(X_train.block(0, i, X_train.rows(), k),
                                                 Y_train.block(0, i, Y_train.rows(), k));
            curI = i+k;
        }
        if (curI != Y_train.cols()){
            split_data_.emplace_back(X_train.block(0, curI, X_train.rows(), X_train.cols() - curI),
                                     Y_train.block(0, curI, Y_train.rows(), Y_train.cols() - curI));
        }
        return split_data_;
    }
public:
    size_t L;
    std::vector<FCLayer*> layers;
    std::vector<std::unordered_map<std::string, md>> caches;
    LossWrapper* loss;
    std::vector<std::pair<md, md>> layers_grads;

    explicit Model(std::vector<FCLayer*>& layers){
        L = layers.size();
        this->layers = layers;
        this->caches = std::vector<std::unordered_map<std::string, md>>(L);
    }

    void add(FCLayer* layer){
        layers.push_back(layer);
    }

    void compile(const std::string& loss,
                 BasicOptimizer* optimizer){
        // TODO add metrics
        this->loss = new LossWrapper{loss};
        this->optimizer = optimizer;
        this->optimizer->init_(layers);
    }

    md forward(const md& X){
        md Y(X);
        for (int i = 0; i < L; ++i){
            Y = layers[i]->forward(Y, caches[i]);
        }
        return Y;
    }

    double get_cost(const md& AL, const md& Y) const{
        return loss->get_cost(AL, Y);
    }

    void backward(const md& AL, const md& Y){
        md dA = loss->get_loss_backward(AL, Y);
        for (int i = L - 1; i >= 0; --i){
            // cache is updated
            dA = layers[i]->backward(dA, caches[i]);
        }
    }

    void update_parameters(int mini_batch_n){
        optimizer->update_parameters(caches, mini_batch_n);

    }

    void fit(const md& X_train, const md& Y_train,
             int epochs=100, bool verbose=false,
             int batch_size=32){
        std::vector<std::pair<md, md>> data_split_ = split_data(X_train, Y_train, batch_size);
        int batches_n = Y_train.cols()/batch_size;
        if (Y_train.cols() % batch_size != 0){
            batches_n++;
        }

        for (int i = 0; i < epochs; ++i){
            md AL;
            md Y;
            for (int b = 0; b < batches_n; ++b){
                md X = data_split_[b].first;
                Y = data_split_[b].second;
                AL = forward(X);
                backward(AL, Y);
                update_parameters(b+1);
            }

            // TODO add metrics
            if (verbose){
                std::cout << "i=" << i << " cost=" <<  get_cost(AL, Y) << std::endl;
            }
        }

    }


    void fit_with_accumulative_gradiant(const md& X_train, const md& Y_train,
             int num_epochs=100, bool verbose=false,
             int mini_batches_num=10){
        std::vector<std::pair<md, md>> data_split_ = split_data(X_train, Y_train, mini_batches_num);

        for (int i = 0; i < num_epochs; ++i){
            // After each backward pass it computes dW, db to cache[l] [0..L)
            // So we need the vector of these caches about each layer also
            layers_grads = std::vector<std::pair<md, md>>(L);
            md AL;
            md Y;
            // TODO add parallelization with OpenMP
            for (int b = 0; b < mini_batches_num; ++b){
                md X = data_split_[b].first;
                Y = data_split_[b].second;
                AL = forward(X);
                backward(AL, Y);
                if (b == 0){
                    for (int l = 0; l < L; ++l){
                        layers_grads[l] = std::make_pair(caches[l]["dW"],
                                                         caches[l]["db"]);
                    }
                }else{
                    for (int l = 0; l < L; ++l){
                        layers_grads[l].first += caches[l]["dW"];
                        layers_grads[l].second += caches[l]["db"];
                    }
                }
            }
            // Accumulative gradient
            for (int l = 0; l < L; ++l){
                caches[l]["dW"] = layers_grads[l].first;
                caches[l]["db"] = layers_grads[l].second;
            }

            update_parameters(1);
            if (verbose){
                std::cout << "i=" << i << " cost=" <<  get_cost(AL, Y) << std::endl;
            }
        }
    }

    md predict(const md& X){
        return forward(X);
    }
};
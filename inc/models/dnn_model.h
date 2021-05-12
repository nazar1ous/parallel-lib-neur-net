#include <iostream>
#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include <Eigen/Core>

#include "layers/config.h"
#include <unordered_map>
#include "layers/loss.h"

class Model{
private:
    std::vector<OptimizerWrapper*> optimizers;
    static std::vector<std::pair<md, md>> split_data(const md& X_train,
                                                     const md& Y_train,
                                                     int mini_batches_num){
        std::vector<std::pair<md, md>> split_data_;
        if (mini_batches_num > Y_train.cols()){
            std::cerr << "Not valid number of mini batches" << std::endl;
            exit(1);
        }
        int k = Y_train.cols()/mini_batches_num;
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

    Model(std::vector<FCLayer*>& layers,
          const std::string& loss_type,
          const std::string& optimizer_type,
        const std::unordered_map<std::string, double>& hparams){

        L = layers.size();
        optimizers = std::vector<OptimizerWrapper*>(L,
                                                    new OptimizerWrapper{optimizer_type, hparams});

        this->layers = layers;
        this->caches = std::vector<std::unordered_map<std::string, md>>(L);
        loss = new LossWrapper{loss_type};
    }

    md forward(const md& X){
        md Y(X);
        for (int i = 0; i < L; ++i){
            Y = layers[i]->forward(Y, caches[i]);
        }
        return Y;
    }

    double get_cost(const md& AL, const md& Y){
        return loss->get_cost(AL, Y);
    }

    void backward(const md& AL, const md& Y){
        md dA = loss->get_loss_backward(AL, Y);
        for (int i = L - 1; i >= 0; --i){
            // cache is updated
            dA = layers[i]->backward(dA, caches[i]);
        }
    }

    void update_parameters(){
        for (int i = 0; i < L; ++i){
            layers[i]->update_params(caches[i], *optimizers[i]);
        }
    }


    void fit(const md& X_train, const md& Y_train,
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

            update_parameters();
            if (verbose){
                std::cout << "i=" << i << " cost=" <<  get_cost(AL, Y) << std::endl;
            }
        }
    }

    md predict(const md& X){
        return forward(X);
    }
};
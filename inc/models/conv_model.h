#ifndef PARALLEL_NEURAL_NET_LIB_CONV_MODEL_H
#define PARALLEL_NEURAL_NET_LIB_CONV_MODEL_H

#include <iostream>
#include "layers/fc_layer.h"
#include "layers/filter.h"
#include "layers/conv_layer.h.h"
#include <Eigen/Dense>
#include <Eigen/Core>

#include "layers/config.h"
#include <unordered_map>
#include "layers/loss.h"

class ConvModel{
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
    std::vector<Conv3D*> layers;
    std::vector<std::unordered_map<std::string, md>> input_caches;
    std::vector<std::unordered_map<std::string, int>> size_caches;
    std::unordered_map<std::string, std::vector<Filter3D>>& filter_caches;
    LossWrapper* loss;

    ConvModel(){}
    //TODO: finish conv model

};

#endif //PARALLEL_NEURAL_NET_LIB_CONV_MODEL_H

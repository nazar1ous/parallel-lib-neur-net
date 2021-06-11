#include <iostream>
//#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h>


#include "layers/config.h"
#include <unordered_map>
#include "layers/loss.h"
#include "layers/optimizers.h"

class Model{
private:
    BasicOptimizer* optimizer{};

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
    size_t L{};
    std::vector<FCLayer*> layers;
    std::vector<std::unordered_map<std::string, md>> caches;
    Loss* loss{};
    std::vector<std::pair<md, md>> layers_grads;

    explicit Model(std::vector<FCLayer*>& layers){
        this->layers = layers;
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
        L = this->layers.size();
        this->caches = std::vector<std::unordered_map<std::string, md>>(L);
    }

    void compile(Loss* loss,
                 BasicOptimizer* optimizer){
        // TODO add metrics
        this->loss = loss;
        this->optimizer = optimizer;
        this->optimizer->init_(layers);
        L = this->layers.size();
        this->caches = std::vector<std::unordered_map<std::string, md>>(L);
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
        // TODO dropout apply mask
        //  Also, if we want L2 regularization, add it to loss 
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

    // The idea of making a bunch of models and then pass batches are stupid because of
    // usage too much memory(each instance of a model has a cache)
    void fit_data_parallel(const md& X_train, const md& Y_train,
                           int epochs=100, bool verbose=false,
                           int batch_size=32,
                           int threads_n=1){

        if (optimizer->type != "sgd"){
            std::cerr << "To use data parallelization, you have to use SGD" << std::endl;
            exit(1);
        }
        std::vector<std::pair<md, md>> data_split_ = split_data(X_train, Y_train, batch_size);
        int batches_n = Y_train.cols()/batch_size;
        if (Y_train.cols() % batch_size != 0){
            batches_n++;
        }

        for (int i = 0; i < epochs; ++i){
            // After each backward pass it computes dW, db to cache[l] [0..L)
            // So we need the vector of these caches about each layer also
            layers_grads = std::vector<std::pair<md, md>>(L);
            md AL;
            md Y;

            omp_set_num_threads(threads_n);
            #pragma omp parallel for default(none) shared(batches_n, data_split_, caches, Y, AL)
            for (int b = 0; b < batches_n; ++b){
                md X = data_split_[b].first;
                Y = data_split_[b].second;
                AL = forward(X);
                backward(AL, Y);
                auto m = X.cols();
                if (b == 0){
                    for (int l = 0; l < L; ++l){
                        layers_grads[l] = std::make_pair(m*caches[l]["dW"],
                                                         m*caches[l]["db"]);
                    }
                }else{
                    for (int l = 0; l < L; ++l){
                        layers_grads[l].first += m*caches[l]["dW"];
                        layers_grads[l].second += m*caches[l]["db"];
                    }
                }
            }
            // Accumulative gradient
            for (int l = 0; l < L; ++l){
                caches[l]["dW"] = layers_grads[l].first/Y_train.cols();
                caches[l]["db"] = layers_grads[l].second/Y_train.cols();
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

    double evaluate(md& X_test, const md& Y_test){
        md Y_hat = predict(X_test);
        md diff = Y_test-Y_hat;
        double good = 0;
        for (int i = 0; i < diff.cols(); ++i){
            if (std::abs(diff(0, i)) <= 0.5){
                good++;
            }
        }
        return good/Y_test.cols();
    }

    Model() {

    }
};
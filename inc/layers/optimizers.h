#include "layers/fc_layer.h"


class BasicOptimizer{
public:
    std::string type;
    std::vector<FCLayer*> layers;
    size_t L;

    virtual void init_(std::vector<FCLayer*>& layers){
        this->layers = layers;
        L = this->layers.size();
    }

    virtual inline void update_param_(md* param, const md& grad, size_t layer_index, int param_index, int mini_batch_n) = 0;

    void update_parameters(std::vector<std::unordered_map<std::string, md>>& caches, int mini_batch_n){
        for (size_t l = 0; l < L; ++l){
            auto params = this->layers[l]->get_params();
            auto grads = this->layers[l]->get_grads();

            for (int i = 0; i < params.size(); ++i){
                auto p = params[i];
                auto grad = grads[i];
                // Update param
                update_param_(p, caches[l][grad], l, i, mini_batch_n);
            }
        }
    };

};


class SGD: public BasicOptimizer{
public:
//    std::string type = "sgd";
    double learning_rate;
    explicit SGD(double learning_rate){
        this->learning_rate = learning_rate;
        this->type = "sgd";
    }

    void update_param_(md* param, const md& grad, size_t layer_index, int param_index, int mini_batch_n) override{
        *param = *param - learning_rate*grad;
    }
};


class GDWithMomentum: public BasicOptimizer{
public:
    double beta_1;
    double learning_rate;
    std::vector<std::vector<md>> V;


    void init_(std::vector<FCLayer*>& layers) override{
        this->layers = layers;
        L = this->layers.size();
        // should be initialized by model before this!
        for (size_t l = 0; l < L; ++l){
            auto layer = layers[l];
            std::vector<md> v;
            for (const auto &param: layer->get_params()){
                md val = md::Zero(param->rows(), param->cols());
                v.push_back(val);
            }
            V.push_back(v);
        }
    }

    GDWithMomentum(double learning_rate, double beta_1=0.9){
        this->learning_rate = learning_rate;
        this->beta_1 = beta_1;
        this->type = "GDWithMomentum";

    }

    void update_param_(md* param, const md& grad, size_t layer_index, int param_index, int mini_batch_n) override{
        V[layer_index][param_index] = beta_1 * V[layer_index][param_index] + (1-beta_1) * grad;
        *param = *param - learning_rate*V[layer_index][param_index];
    }
};


class RMSprop: public BasicOptimizer{
public:
    double beta_2;
    double epsilon;
    double learning_rate;
    std::vector<std::vector<md>> S;


    void init_(std::vector<FCLayer*>& layers) override{
        this->layers = layers;
        L = this->layers.size();
        // should be initialized by model before this!
        for (size_t l = 0; l < L; ++l){
            auto layer = layers[l];
            std::vector<md> v;
            for (const auto &param: layer->get_params()){
                md val = md::Zero(param->rows(), param->cols());
                v.push_back(val);
            }
            S.push_back(v);
        }
    }

    RMSprop(double learning_rate, double beta_2=0.999, double epsilon=1e-8){
        this->learning_rate = learning_rate;
        this->beta_2 = beta_2;
        this->epsilon = epsilon;
        this->type = "RMSprop";
    }

    void update_param_(md* param, const md& grad, size_t layer_index, int param_index, int mini_batch_n) override{
        S[layer_index][param_index] = beta_2 * S[layer_index][param_index] + (1-beta_2) * grad.array().square().matrix();
        *param = *param - (learning_rate*grad.array()/(S[layer_index][param_index].array().sqrt()+epsilon)).matrix();
    }
};


class Adam: public BasicOptimizer{
public:
    double beta_1;
    double beta_2;
    double epsilon;
    double learning_rate;
    std::vector<std::vector<md>> S;
    std::vector<std::vector<md>> V;

    void init_(std::vector<FCLayer*>& layers) override{
        this->layers = layers;
        L = this->layers.size();
        // should be initialized by model before this!
        for (size_t l = 0; l < L; ++l){
            auto layer = layers[l];
            std::vector<md> v;
            for (const auto &param: layer->get_params()){
                md val = md::Zero(param->rows(), param->cols());
                v.push_back(val);
            }
            S.push_back(v);
        }
        for (size_t l = 0; l < L; ++l){
            auto layer = layers[l];
            std::vector<md> v;
            for (const auto &param: layer->get_params()){
                md val = md::Zero(param->rows(), param->cols());
                v.push_back(val);
            }
            V.push_back(v);
        }
    }

    Adam(double learning_rate, double beta_1=0.9, double beta_2=0.999, double epsilon=1e-8){
        this->learning_rate = learning_rate;
        this->beta_1 = beta_1;
        this->beta_2 = beta_2;
        this->epsilon = epsilon;
        this->type = "Adam";
    }

    void update_param_(md* param, const md& grad, size_t layer_index, int param_index, int mini_batch_n) override{
        V[layer_index][param_index] = beta_1 * V[layer_index][param_index] + (1-beta_1) * grad;
        S[layer_index][param_index] = beta_2 * S[layer_index][param_index] + (1-beta_2) * grad.array().square().matrix();
        auto V_corr = V[layer_index][param_index]/(1-std::pow(beta_1, mini_batch_n));
        auto S_corr = S[layer_index][param_index]/(1-std::pow(beta_2, mini_batch_n));

        *param = *param - learning_rate*(V_corr.array()/(S_corr.array().sqrt() + epsilon)).matrix();
    }
};

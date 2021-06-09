#include "layers/config.h"


class BasicOptimizer{
public:
    std::string type;
    std::vector<md*> params;
    std::vector<md*> grads;

    virtual void init_(const std::vector<md*>& params,
                       const std::vector<md*>& grads){
        this->params = params;
        this->grads = grads;

    }

    virtual std::vector<std::vector<md*>> get_matrices_def() {
        return std::vector<std::vector<md*>>{std::ref(params), std::ref(grads)};
    }

    virtual std::vector<std::vector<md>> get_matrices_ndef() {
        return std::vector<std::vector<md>>{};
    }

    virtual std::vector<double> get_hparams_(){
        return std::vector<double>{};
    }


    virtual void update_param_(md* param, const md& grad, int param_index, int mini_batch_n){};

    void update_parameters(int mini_batch_n){

        for (int i = 0; i < params.size(); ++i){
            auto p = params[i];
            auto grad = *(grads[i]);
            // Update param
            update_param_(p, grad, i, mini_batch_n);
        }
    };

    BasicOptimizer() = default;


};


class SGD: public BasicOptimizer{
public:
    double learning_rate;
    explicit SGD(double learning_rate){
        this->learning_rate = learning_rate;
        this->type = "sgd";
    }

    void update_param_(md* param, const md& grad, int param_index, int mini_batch_n) override{
        *param = *param - learning_rate*grad;
    }


    std::vector<double> get_hparams_() override{
        return std::vector<double>{std::ref(learning_rate)};
    }
};


class GDWithMomentum: public BasicOptimizer{
public:
    double beta_1;
    double learning_rate;
    std::vector<md> V;

    void init_(const std::vector<md*>& params,
               const std::vector<md*>& grads) override{
        this->params = params;
        this->grads = grads;
        for (const auto &param: this->params){
            md val = md::Zero(param->rows(), param->cols());
            V.push_back(val);
        }
    }

    explicit GDWithMomentum(double learning_rate, double beta_1=0.9){
        this->learning_rate = learning_rate;
        this->beta_1 = beta_1;
        this->type = "GDWithMomentum";

    }

    void update_param_(md* param, const md& grad, int param_index, int mini_batch_n) override{
        V[param_index] = beta_1 * V[param_index] + (1-beta_1) * grad;
        *param = *param - learning_rate*V[param_index];
    }

    std::vector<std::vector<md>> get_matrices_ndef() override{
        return std::vector<std::vector<md>>{std::ref(V)};
    }

    std::vector<double> get_hparams_() override{
        return std::vector<double>{std::ref(learning_rate), std::ref(beta_1)};
    }
};


class RMSprop: public BasicOptimizer{
public:
    double beta_2;
    double epsilon;
    double learning_rate;
    std::vector<md> S;


    void init_(const std::vector<md*>& params,
               const std::vector<md*>& grads) override{
        this->params = params;
        this->grads = grads;
        for (const auto &param: this->params){
            md val = md::Zero(param->rows(), param->cols());
            S.push_back(val);
        }
    }

    explicit RMSprop(double learning_rate, double beta_2=0.999, double epsilon=1e-8){
        this->learning_rate = learning_rate;
        this->beta_2 = beta_2;
        this->epsilon = epsilon;
        this->type = "RMSprop";
    }

    void update_param_(md* param, const md& grad, int param_index, int mini_batch_n) override{
        S[param_index] = beta_2 * S[param_index] + (1-beta_2) * grad.array().square().matrix();
        *param = *param - (learning_rate*grad.array()/(S[param_index].array().sqrt()+epsilon)).matrix();
    }


    std::vector<std::vector<md>> get_matrices_ndef() override{
        return std::vector<std::vector<md>>{std::ref(S)};
    }

    std::vector<double> get_hparams_() override{
        return std::vector<double>{std::ref(learning_rate), std::ref(beta_2), std::ref(epsilon)};
    }
};


class Adam: public BasicOptimizer{
public:
    double beta_1;
    double beta_2;
    double epsilon;
    double learning_rate;
    std::vector<md> S;
    std::vector<md> V;

    void init_(const std::vector<md*>& params,
               const std::vector<md*>& grads) override{
        this->params = params;
        this->grads = grads;

        for (const auto &param: this->params){
            md val = md::Zero(param->rows(), param->cols());
            S.push_back(val);
            V.push_back(val);
        }
    }

    explicit Adam(double learning_rate, double beta_1=0.9, double beta_2=0.999, double epsilon=1e-8){
        this->learning_rate = learning_rate;
        this->beta_1 = beta_1;
        this->beta_2 = beta_2;
        this->epsilon = epsilon;
        this->type = "Adam";
    }

    void update_param_(md* param, const md& grad, int param_index, int mini_batch_n) override{
        V[param_index] = beta_1 * V[param_index] + (1-beta_1) * grad;
        S[param_index] = beta_2 * S[param_index] + (1-beta_2) * grad.array().square().matrix();
        auto V_corr = V[param_index]/(1-std::pow(beta_1, mini_batch_n));
        auto S_corr = S[param_index]/(1-std::pow(beta_2, mini_batch_n));

        *param = *param - learning_rate*(V_corr.array()/(S_corr.array().sqrt() + epsilon)).matrix();
    }


    std::vector<std::vector<md>> get_matrices_ndef() override{
        return std::vector<std::vector<md>>{std::ref(V), std::ref(S)};
    }

     std::vector<double> get_hparams_() override{
        return std::vector<double>{std::ref(learning_rate),
                                    std::ref(beta_1),
                                    std::ref(beta_2),
                                    std::ref(epsilon)};
    }
};

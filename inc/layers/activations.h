#ifndef PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H
#define PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "layers/config.h"

class Activation{

public:
    std::string type;
    virtual md activate_forward(const md& x){
        return md{};
    };
    virtual md activate_backward(const md& dA,
                                              const md& Z){
        return md{};
    };
};

class Sigmoid: public Activation{
public:
    std::string type = "sigmoid";
    md activate_forward(const md& x) override{
        return 1 / ((-x.array()).exp() + 1);
    }
    md activate_backward(const md& dA, const md& Z) override{
        md s = activate_forward(Z);
        return dA.array() * s.array() * (1 - s.array());
    }
};


class Linear: public Activation{
public:
    std::string type = "linear";
    md activate_forward(const md& x) override{
        return md (x);
    }
    md activate_backward(const md& dA, const md& Z) override{
        return md (dA);
    }
};

class ReLu: public Activation{
public:
    std::string type = "relu";
    md activate_forward(const md& x) override{
        return x.array().cwiseMax(0);
    }
    md activate_backward(const md& dA, const md& Z) override{
        return dA.array() * Z.unaryExpr([](double y){return (double)(y>0);}).array();
    }
};

class Tanh: public Activation{
public:
    std::string type = "tanh";
    md activate_forward(const md& x) override{
        return x.array().tanh();
    }
    md activate_backward(const md& dA, const md& Z) override{
        return dA.array() * (1 - Z.array().tanh().square());
    }
};


class SoftMax: public Activation{
public:
    std::string type = "softmax";
    md activate_forward(const md& x) override{
        auto s = x.array().exp();
        return s/s.sum();
    }
    md activate_backward(const md& dA, const md& Z) override{
        return md(dA);
    }
};

class ActivationWrapper{
public:
    Activation wrapper;
    std::string type;
    void builder(){

        if (this->type == "sigmoid"){
            wrapper = Sigmoid{};
        }else if (this->type == "relu"){
            wrapper = ReLu{};
        }else if (this->type == "linear"){
            wrapper = Linear{};
        }else if (this->type == "tanh") {
            wrapper = Tanh{};
        }else if (this->type == "softmax") {
            wrapper = SoftMax{};
        }else{
            std::cerr << "Not implemented type of activation";
            exit(1);
        }
    }
    explicit ActivationWrapper(const std::string& type){
        this->type = boost::to_lower_copy(type);
        builder();

    }

    md activate_forward(const md& x) {
        return wrapper.activate_forward(x);
    }

    md activate_backward(const md& dA,
                                      const md& Z) {
        return wrapper.activate_backward(dA, Z);
    }

    ActivationWrapper() = default;

};


#endif //PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H

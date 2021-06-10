#ifndef PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H
#define PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "layers/config.h"

class Activation{

public:
    std::string type;
    virtual inline md activate_forward(const md& x) = 0;
    virtual inline md activate_backward(const md& dA,
                                        const md& Z) = 0;

    Activation(const Activation& copy){
        this->type = copy.type;
    }

    Activation() = default;
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

    Sigmoid(const Sigmoid& copy) : Activation(copy) {
        this->type = copy.type;
    }
    Sigmoid() = default;

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
    Linear(const Linear& copy) : Activation(copy) {
        this->type = copy.type;
    }
    Linear() = default;

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
    ReLu(const ReLu& copy) : Activation(copy) {
        this->type = copy.type;
    }
    ReLu() = default;

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
    Tanh(const Tanh& copy) : Activation(copy) {
        this->type = copy.type;
    }
    Tanh() = default;

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
    SoftMax() = default;
    SoftMax(const SoftMax& copy) : Activation(copy) {
        this->type = copy.type;
    }
};

class ActivationWrapper: public Activation{
private:
    Activation *wrapper;
    void builder(){
        auto value = boost::to_lower_copy(this->type);

        if (value == "sigmoid"){
            wrapper = new Sigmoid{};
        }else if (value == "relu"){
            wrapper = new ReLu{};
        }else if (value == "linear"){
            wrapper = new Linear{};
        }else if (value == "tanh") {
            wrapper = new Tanh{};
        }else if (value == "softmax") {
            wrapper = new SoftMax{};
        }else{
            std::cerr << "Not implemented type of activation";
            exit(1);
        }
    }
public:
    explicit ActivationWrapper(const std::string& type){
        this->type = type;
        builder();
    }
    md activate_forward(const md& x) override {
        return wrapper->activate_forward(x);
    }

    md activate_backward(const md& dA,
                         const md& Z) override {
        return wrapper->activate_backward(dA, Z);
    }

    ActivationWrapper(const ActivationWrapper& copy) : Activation(copy) {
        this->wrapper = copy.wrapper;
        this->type = copy.type;
    }

};


#endif //PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H

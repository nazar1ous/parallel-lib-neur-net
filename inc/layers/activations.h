#ifndef PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H
#define PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H
#include <Eigen/Core>
#include <Eigen/Dense>


template<typename T>
class Activation{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;

public:
    std::string type;
    virtual inline MatrixTx activate_forward(const MatrixTx& x) = 0;
    virtual inline MatrixTx activate_backward(const MatrixTx& dA,
                                              const MatrixTx& Z) = 0;
};

template<typename T>
class Sigmoid: public Activation<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type = "sigmoid";
    inline MatrixTx activate_forward(const MatrixTx& x) override{
        return T(1) / ((-x.array()).exp() + T(1));
    }
    inline MatrixTx activate_backward(const MatrixTx& dA, const MatrixTx& Z) override{
        auto s = activate_forward(Z);
        return dA.array() * s.array() * (T(1) - s.array());
    }
};


template<typename T>
class Linear: public Activation<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type = "linear";
    inline MatrixTx activate_forward(const MatrixTx& x) override{
        return std::copy(x);
    }
    inline MatrixTx activate_backward(const MatrixTx& dA, const MatrixTx& Z) override{
        return std::copy(dA);
    }
};

template<typename T>
class ReLu: public Activation<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type = "relu";
    inline MatrixTx activate_forward(const MatrixTx& x) override{
        return x.array().cwiseMax(T(0));
    }
    inline MatrixTx activate_backward(const MatrixTx& dA, const MatrixTx& Z) override{
        return dA.array() * Z.unaryExpr([](T y){return (T)(y>0);});
    }
};

template<typename T>
class Tanh: public Activation<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type = "tanh";
    inline MatrixTx activate_forward(const MatrixTx& x) override{
        return x.array().tanh();
    }
    inline MatrixTx activate_backward(const MatrixTx& dA, const MatrixTx& Z) override{
        return dA.array() * (T(1) - Z.array().tanh().square());
    }
};


template<typename T>
class ActivationWrapper: public Activation<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
private:
    Activation<T> wrapper;
    void builder(){
        auto value = std::tolower(this->type);
        if (value == "sigmoid"){
            wrapper = new Sigmoid<T>{};
        }else if (value == "relu"){
            wrapper = new ReLu<T>{};
        }else if (value == "linear"){
            wrapper = new Linear<T>{};
        }else if (value == "tanh") {
            wrapper = new Tanh<T>{};
        }else{
                std::cerr << "Not implemented type of activation";
            }
    }
public:
    explicit ActivationWrapper(const std::string& type){
        this->type = type;
        builder();
    }
    inline MatrixTx activate_forward(const MatrixTx& x) override {
        return wrapper.activate_forward(x);
    }

    inline MatrixTx activate_backward(const MatrixTx& dA,
                                      const MatrixTx& Z) override {
        return wrapper.activate_backward(dA, Z);
    }

};


#endif //PARALLEL_NEURAL_NET_LIB_ACTIVATIONS_H

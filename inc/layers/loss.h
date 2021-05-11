#include "layers/config.h"


class Loss{
public:
    std::string type;
    virtual double inline get_loss(const md& AL, const md& Y) = 0;
    virtual md inline get_loss_backward(const md& AL, const md& Y) = 0;
};

class MSE: public Loss{
public:
    type = "mse";

    double get_loss(const md& AL, const md& Y) override{
        return (Y-AL).array().square().colwise().sum()/Y.cols();
    }

    md get_loss_backward(const md& AL, const md& Y) override{
        return -2 * (Y - AL);
    }
};


class BinaryCrossEntropy: public Loss{
public:
    type = "binary_cross_entropy";

    double get_loss(const md& AL, const md& Y) override{
        return AL.array().log() * Y.array() + (1 - AL.array()).log() * (1 - Y.array());
    }

    md get_loss_backward(const md& AL, const md& Y) override{
        return - ((Y.array()/AL.array() - (1 - Y.array())/(1 - AL.array())));
    }
};


class LossWrapper: public Loss{
private:
    Loss* wrapper;
    void builder(){
        auto value = this->type;
        if (value == "mse"){
            wrapper = new MSE{};
        }else if (value == "binary_cross_entropy"){
            wrapper = new BinaryCrossEntropy{};
        }else{
            std::cerr << "Not implemented type of Loss function";
        }
    }
public:

    explicit LossWrapper(const std::string& type){
        this->type = boost::to_lower_copy(type);
        builder();
    }

    double get_loss(const md& AL, const md& Y) override{
        return wrapper->get_loss(AL, Y);
    }

    md get_loss_backward(const md& AL, const md& Y) override{
        return wrapper->get_loss_backward(AL, Y);
    }
};
#include <Eigen/Dense>
#include <iostream>
#include "layers/config.h"


class BasicOptimizer{
public:
    std::string type;
    std::unordered_map<std::string, double> hparams;
    std::unordered_map<std::string, md> matrix_cache;
    std::unordered_map<std::string, double> default_cache;
    explicit BasicOptimizer(const std::string& type,
                            const std::unordered_map<std::string, double>& hparams){
        this->hparams = std::ref(hparams);
        this->type = type;
    }

    virtual inline void update_parameters(md* W, md* b,
                                          std::unordered_map<std::string, md>& cache) = 0;
};


class GDOptimizer: public BasicOptimizer{
public:
    std::string type = "gd";
    GDOptimizer(const std::string& type,
                const std::unordered_map<std::string, double>& hparams): BasicOptimizer(type, hparams){

    }

    void update_parameters(md* W, md* b,
                           std::unordered_map<std::string, md>& cache) override{
        *W = *W - this->hparams["alpha"]*cache["dW"];
        *b = *b - this->hparams["alpha"]*cache["db"];
    };

};


class OptimizerWrapper: public BasicOptimizer{
private:
    BasicOptimizer *wrapper;


public:
    void builder(){
        if (this->type == "gd"){
            wrapper = new GDOptimizer{this->type, this->hparams};
        }else{
            std::cerr << "Not implemented type of optimizer\n";
        }
        if (this->hparams.find("alpha") == this->hparams.end()){
            std::cerr << "There is no alpha param for GD optimizer\n";
        }
    }

    OptimizerWrapper(const std::string& type,
                     const std::unordered_map<std::string, double>& hparams): BasicOptimizer(type, hparams){
        builder();
    }

    void update_parameters(md* W, md* b,
                           std::unordered_map<std::string, md>& cache) override{
        return wrapper->update_parameters(W, b, cache);
    }
};
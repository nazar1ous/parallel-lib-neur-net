#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


template<typename T>
class BasicOptimizer{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
private:
    std::unordered_map<std::string, T> hparams;
    std::unordered_map<std::string, MatrixTx> matrix_cache;
    std::unordered_map<std::string, T> default_cache;

public:
    explicit BasicOptimizer(const std::unordered_map<std::string, T>& hparams){
        this->hparams = std::ref(hparams);
    }

    virtual inline void update_parameters(MatrixTx* W, MatrixTx* b,
                                          const std::unordered_map<std::string, MatrixTx>& cache) = 0;
};

template<typename T>
class GDOptimizer: private BasicOptimizer<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type = "gd";
    void update_parameters(MatrixTx* W, MatrixTx* b,
                           const std::unordered_map<std::string, MatrixTx>& cache) override{
        if (!this->hparams.find("alpha")){
            std::cerr << "There is no alpha param for GD optimizer\n";
        }
        *W = *W - this->hparams["alpha"]*cache["dW"];
        *b = *b - this->hparams["alpha"]*cache["db"];
    };

};


template<typename T>
class OptimizerWrapper: private BasicOptimizer<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
private:
    std::string type;
    BasicOptimizer<T> wrapper;


public:
    OptimizerWrapper(const std::string& type,
                     std::unordered_map<std::string, T> hparams) override{
        this->hparams = hparams;
        this->type = type;
    }

    void builder(){
        if (type == "gd"){
            wrapper = GDOptimizer<T>{this->hparams};
        }else{
            std::cerr << "Not implemented type of optimizer\n";
        }
    }
    explicit OptimizerWrapper(const std::unordered_map<std::string, T>& hparams){
        this->hparams = std::ref(hparams);
        builder();
    }

    void update_parameters(MatrixTx* W, MatrixTx* b,
                           const std::unordered_map<std::string, MatrixTx>& cache) override{
        return wrapper.update_parameters(W, b, cache);
    }
};
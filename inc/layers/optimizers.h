#include <Eigen/Dense>
#include <iostream>


template<typename T>
class BasicOptimizer{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type;
    std::unordered_map<std::string, T> hparams;
    std::unordered_map<std::string, MatrixTx> matrix_cache;
    std::unordered_map<std::string, T> default_cache;
    explicit BasicOptimizer(const std::string& type,
                            const std::unordered_map<std::string, T>& hparams){
        this->hparams = std::ref(hparams);
        this->type = type;
    }

    virtual inline void update_parameters(MatrixTx* W, MatrixTx* b,
                                          std::unordered_map<std::string, MatrixTx>& cache) = 0;
};

template<typename T>
class GDOptimizer: public BasicOptimizer<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    std::string type = "gd";
    GDOptimizer(const std::string& type,
                const std::unordered_map<std::string, T>& hparams): BasicOptimizer<T>(type, hparams){

    }

    void update_parameters(MatrixTx* W, MatrixTx* b,
                           std::unordered_map<std::string, MatrixTx>& cache) override{
        *W = *W - this->hparams["alpha"]*cache["dW"];
        *b = *b - this->hparams["alpha"]*cache["db"];
    };

};


template<typename T>
class OptimizerWrapper: public BasicOptimizer<T>{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
private:
    BasicOptimizer<T> *wrapper;


public:
    void builder(){
        if (this->type == "gd"){
            wrapper = new GDOptimizer<T>{this->type, this->hparams};
        }else{
            std::cerr << "Not implemented type of optimizer\n";
        }
        if (this->hparams.find("alpha") == this->hparams.end()){
            std::cerr << "There is no alpha param for GD optimizer\n";
        }
    }

    OptimizerWrapper(const std::string& type,
                     const std::unordered_map<std::string, T>& hparams): BasicOptimizer<T>(type, hparams){
        builder();
    }

    void update_parameters(MatrixTx* W, MatrixTx* b,
                           std::unordered_map<std::string, MatrixTx>& cache){
        return wrapper->update_parameters(W, b, cache);
    }
};
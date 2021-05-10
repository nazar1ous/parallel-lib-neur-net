#include <iostream>
#include "layers/fc_layer.h"
#include <Eigen/Dense>
#include "layers/config.h"

template<typename T>
class Model{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    int L;
    std::vector<FCLayer<T>*> layers;
    std::vector<std::unordered_map<std::string, MatrixTx>> caches;
    // TODO probably put the optimizer here
    //  that is common to all layers
    explicit Model(std::vector<FCLayer<T>*>& layers){
        L = layers.size();
        this->layers = layers;
        caches.reserve(L);
    }

    MatrixTx forward(const MatrixTx& X){
        // TODO fucking bug with code 136

        MatrixTx Y = X.replicate(X.rows(), X.cols());
        //
        for (int i = 0; i < L; ++i){
            Y = layers[i]->forward(Y, caches[i]);
        }
        return Y;
    }

    void backward(const MatrixTx& AL, const MatrixTx& Y){
        // TODO add different losses
        MatrixTx dA = - ((Y.array()/AL.array() - (T(1) - Y.array())/(T(1) - AL.array())));
        for (int i = L - 1; i >= 0; --i){
            // cache is updated
            dA = layers[i]->backward(dA, caches[i]);
        }
    }

    void update_parameters(){
        for (int i = 0; i < L; ++i){
            layers[i]->update_params(caches[i]);
        }
    }

    void fit(const MatrixTx& X_train, const MatrixTx& Y_train,
             int max_iter=1000){
        for (int i = 0; i < max_iter; ++i){
            auto AL = forward(X_train);
            std::cout << "h1";
            backward(AL, Y_train);
            std::cout << "h2";

            update_parameters();
            std::cout << "h3";

        }
    }

    void predict(const MatrixTx& X){
        return forward(X);
    }
};
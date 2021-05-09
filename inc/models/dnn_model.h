#include <iostream>
#include "layers/fc_layer.h"

template<typename T>
class Model{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;
public:
    int L;
    std::vector<FCLayer<T>*>& layers;
    std::vector<std::unordered_map<std::string, MatrixTx>> caches;
    // TODO probably put the optimizer here
    //  that is common to all layers
    explicit Model(std::vector<FCLayer<T>*>& layers){
        L = layers.size();
        this->layers = std::ref(layers);
        caches.reserve(L);
    }

    MatrixTx forward(const MatrixTx& X){
        MatrixTx Y = X;
        for (int i = 0; i < L; ++i){
            Y = layers[i].forward(Y, caches[i]);
        }
        return Y;
    }

    void backward(const MatrixTx& AL, const MatrixTx& Y){
        // TODO add different losses
        MatrixTx dA = - ((Y.array()/AL.array() - (T(1) - Y)/(T(1) - AL)));
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
            backward(AL, Y_train);
            update_parameters();
        }
    }

    void predict(const MatrixTx& X){
        return forward(X);
    }
};
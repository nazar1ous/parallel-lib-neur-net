#include "../inc/linalg_1/matrix.hpp"

template<typename T>
void Matrix2D<T>::print() const{
    std::cout << "[\n";
    for (int i = 0; i < rows_n; ++i){
        std::cout << "[";
        for (int j = 0; j < cols_n; ++j){
            std::cout << get(i, j) << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}
template<typename T>
T Matrix2D<T>::get(size_t i_row, size_t j_col) const{
    return arr[i_row*cols_n + j_col];
}
template<typename T>
void Matrix2D<T>::set(size_t i, size_t j, T val){
    arr[i*cols_n + j] = val;
}

// TODO parallel computing
template<typename T>
void Matrix2D<T>::apply(T (*func)(T), Matrix2D<T>* res) {
    for (int i = 0; i < rows_n; ++i){
        for (int j = 0; j < cols_n; ++j){
            res->set(i, j, func(get(i, j)));
        }
    }
}

// TODO parallel computing
template<typename T>
void Matrix2D<T>::mul(const Matrix2D<T> &first, const Matrix2D<T>& second,
                      Matrix2D<T>* res) {
    if (first.cols_n != second.rows_n)
        exit(1);
    int n = first.row_n,
    m = first.col_n,
    p = second.col_n;
    int i,j,k;

    #pragma omp parallel shared(A,B,C) private(i,j,k)
    {
        #pragma omp for  schedule(static)
        for (i = 0; i < m; ++i) {
            // Multiply with each column of B
            for (j = 0; j < p; ++j) {

                for (k = 0; k < n; ++k) {
                    res->set(i, j) += first.get(i, k) * second.get(k, j);
                }
            }
        }
    }
}

template<typename T>
Matrix2D<T>* Matrix2D<T>::tr(){
    Matrix2D<T> m(cols_n, rows_n);
    for (int i = 0; i < rows_n; ++i){
        for (int j = 0; j < cols_n; ++j){
            m.set(j, i) = this->get(i, j);
        }
    }
    return *m;
}

template<typename T>
void Matrix2D<T>::plus(const Matrix2D<T>& other, Matrix2D<T>* res){
    for (int i = 0; i < this->rows_n; ++i){
        for (int j = 0; j < this->cols_n; ++j){
            res->set(i, j, this->get(i, j)+other.get(i, j));
        }
    }
}

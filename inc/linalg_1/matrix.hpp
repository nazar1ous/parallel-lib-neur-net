
#ifndef ADDER_MATRIX_H
#define ADDER_MATRIX_H

#include <iostream>


template<typename T>
class Matrix2D{
private:
    T* arr;
public:
    size_t rows_n = 0, cols_n = 0;
    Matrix2D(size_t rows_n, size_t cols_n){
        this->rows_n = rows_n;
        this->cols_n = cols_n;
        arr = new T[rows_n*cols_n];

    }
    void init(T init_val){
        for (int i = 0; i < rows_n*cols_n; ++i){
            arr[i] = init_val;
        }
    }

    T get(size_t i_row, size_t j_col) const;

    void print() const;

    void set(size_t i, size_t j, T val);

    void apply(T (*func)(T), Matrix2D<T>* res);

    Matrix2D<T>* tr();

    static void mul(const Matrix2D<T>& first, const Matrix2D<T>& second,
                    Matrix2D<T>* res);

    void plus(const Matrix2D<T>& other, Matrix2D<T>* res);

    Matrix2D<T> operator+(const Matrix2D<T>& other);
    Matrix2D<T> operator*(const Matrix2D<T>& other);
    Matrix2D<T> operator/(const Matrix2D<T>& other);
    Matrix2D<T> operator-(const Matrix2D<T>& other);
    Matrix2D<T> operator*(const T& val);
    Matrix2D<T> operator+(const T& val);
    Matrix2D<T> operator-(const T& val);
    Matrix2D<T> operator/(const T& val);





    ~Matrix2D(){
        delete []arr;
    }
};
#endif //ADDER_MATRIX_H

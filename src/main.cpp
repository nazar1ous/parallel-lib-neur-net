// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

// Remember to include ALL the necessary headers!
#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>

// By convention, C++ header files use the `.hpp` extension. `.h` is OK too.
#include "arithmetic/arithmetic.hpp"
#include <vector>
#include <bits/stdc++.h>

class Matrix2D{
public:
    // m - number of rows, n - number of columns
    int row_n, col_n;
    std::vector<std::vector<double>> _arr;

    Matrix2D(int m, int n, int init_val=0){
        using namespace std;
        this->row_n = m; this->col_n = n;
        vector<vector<double>> vec(m , vector<double> (n, init_val));
        this->_arr = vec;
    }



    // Multiply matrix A by B and put in the result in C
    // A (m, n), B(n, p)
    static void _multiply(const Matrix2D& A, const Matrix2D& B, Matrix2D &C){
        int n = A.row_n,
        m = A.col_n,
        p = B.col_n;
        int i,j,k;
        #pragma omp parallel shared(A,B,C) private(i,j,k)
        {
            #pragma omp for  schedule(static)
            for (i = 0; i < m; ++i) {
                // Multiply with each column of B
                for (j = 0; j < p; ++j) {

                    for (k = 0; k < n; ++k) {
                        C._arr[i][j] += A._arr[i][k] * B._arr[k][j];
                    }
                }
            }
        }
    }

    void print(){
        for (int i = 0; i < row_n; ++i){
            std::cout << "[";
            for (int j = 0; j < col_n; ++j){
                std::cout << std::to_string(this->_arr[i][j]) << ",";
            }
            std::cout << "],\n";
        }
    }

};


#define N 3
#define P 3
#define M 3

int main(int argc, char **argv) {
    auto a = new Matrix2D(M, N),
    b = new Matrix2D(N, P),
    c = new Matrix2D(M, P);

    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            a->_arr[i][j] = i+j;

        }
    }
    a->print();
    c->print();

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < P; ++j){
            b->_arr[i][j] = i+j;
        }
    }
    c->_multiply(*a, *b, *c);
    c->print();

//    shit(c);
//    alg_matmul2D(N, P, M, c, a, b);

}

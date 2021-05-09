#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
//#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "layers/fc_layer.h"
#include "layers/activations.h"





int main(int argc, char **argv) {
    std::map<std::string, int> m = {std::pair{"shit", 2},
                                    std::pair{"gavno", 3}};
    std::cout << m["gavno"];

}

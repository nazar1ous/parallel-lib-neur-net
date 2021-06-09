#include <boost/mpi.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <sstream>
#include <iomanip>
#include "layers/config.h"

class mdWrapper{
private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, unsigned ){
        ar & matrix;
    }

public:
    explicit mdWrapper(const md& matrix){
        this->matrix = matrix;
    }
    md matrix;
};
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
//#include "layers/structures_mpi.h"
//#include "models/dnn_model.h"
#include "layers/config.h"
#include <fstream>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include "models/dnn_model_mpi.h"

//#include "layers/filter.h"



namespace boost { namespace serialization {

        template<class Archive>
        inline void serialize(Archive & ar,
                              md & matrix,
                              const unsigned int version)
        {
            size_t rows = matrix.rows();
            size_t cols = matrix.cols();
            ar & make_nvp("rows", rows);
            ar & make_nvp("cols", cols);
            matrix.resize(rows, cols);
            for(int r = 0; r < rows; ++r)
                for(int c = 0; c < cols; ++c)
                    ar & make_nvp("val", matrix(r,c));
        }

        template<class Archive>
        inline void serialize(Archive & ar,
                              Activation & act,
                              const unsigned int version)
        {
            ar & act.type;
        }

        template<class Archive>
        inline void serialize(Archive & ar,
                              ActivationWrapper & act,
                              const unsigned int version)
        {
            ar & act.type;
            ar & act.wrapper;
        }
        template<class Archive>
        inline void serialize(Archive & ar,
                              std::vector<md> & vec,
                              const unsigned int version)
        {
            for (auto &item: vec){
                ar & item;
            }
        }

        template<class Archive>
        inline void serialize(Archive & ar,
                              std::vector<md*> & vec,
                              const unsigned int version)
        {
            for (auto &item: vec){
                ar & item;
            }
        }

        template<class Archive>
        inline void serialize(Archive & ar,
                              BasicOptimizer & opt,
                              const unsigned int version)
        {
            ar & opt.type;
            std::vector<std::vector<md *>> layers_params = opt.get_matrices_def();
            std::vector<std::vector<md>> opt_caches = opt.get_matrices_ndef();
            std::vector<double> opt_hparams = opt.get_hparams_();

            for ( auto& vect_param: layers_params){
                ar & vect_param;
            }

            for ( auto& vect_cache: opt_caches){
                ar & vect_cache;
            }

            for ( auto& hparam: opt_hparams){
                ar & hparam;
            }
        }



        template<class Archive>
        inline void serialize(Archive & ar,
                              FCLayer & layer,
                              const unsigned int version)
        {
            ar & layer.W;
            ar & layer.b;
            ar & layer.A_prev;
            ar & layer.Z;
            ar & layer.dW;
            ar & layer.db;
            ar & layer.input_size;
            ar & layer.output_size;
            ar & layer.stddev;
            ar & layer.initialization;
            ar & layer.activation;
            ar & layer.optimizer;

        }

    }} // namespace boost::serialization

    
namespace mpi = boost::mpi;

std::pair<md, md> read_file_data(std::fstream *in_file){
    std::string temp;
    *in_file >> temp;
    auto n_x = std::stoi(temp);
    *in_file >> temp;
    auto m = std::stoi(temp);
    md X(n_x, m);
    md Y(1, m);

    for (int sample = 0; sample < m; sample++){
        *in_file >> temp;
        Y(0, sample) = std::stod(temp);
        for (int x = 0; x < n_x; ++x){
            *in_file >> temp;
            X(x, sample) = std::stod(temp);
        }
    }

    return std::make_pair(X, Y);
}

std::vector<md> pre_process_data(){

    std::fstream in_file("/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/tests/gen_data/cpp_dataset.txt");
    auto data = read_file_data(&in_file);
    md X = data.first;
    md Y = data.second;
    int diff = X.cols() * 0.15;

    md X_train = X.block(0, 0, X.rows(), X.cols() - diff)/255;
    md Y_train = Y.block(0, 0, Y.rows(), Y.cols() - diff);

    md X_test = X.block(0, X.cols() - diff, X.rows(), diff)/255;
    md Y_test = Y.block(0, Y.cols() - diff, Y.rows(), diff);
    return std::vector{X_train, X_test, Y_train, Y_test};
}


void test_fc_layer_mpi(int argc, char *argv[]){
    mpi::environment env(argc, argv);
    mpi::communicator world;
    bool is_generator = world.rank() > 0;
    mpi::communicator slaves = world.split(is_generator? 1: 0);
    md m;
    FCLayer layer;
    int epochs;
    std::vector<FCLayer> layers;
    bool stage_identifier_forward = true;

    if (world.rank() == 0){
        // get data
        auto data = pre_process_data();
        auto X_train = data[0];
        auto X_test = data[1];
        auto Y_train = data[2];
        auto Y_test = data[3];

        // initialize hyper parameters
        // L - number of layers
        int L = 4;
        double learning_rate = 0.01;
        epochs = 1;
        int batch_size = 32;
        std::vector<BasicOptimizer> ops(L);
        for (int l = 0; l < L; ++l){
            ops[l] = SGD(learning_rate);
        }
        std::string loss_type = "BinaryCrossEntropy";


        // specify layers
        auto l1 =  FCLayer(X_train.rows(), 10, "linear", ops[0], "he", 1);
        auto l2 = FCLayer(10, 20, "tanh", ops[1], "he",1);
        auto l3 = FCLayer(20, 20, "relu",ops[2], "he",1);
        auto l4 =  FCLayer(20, Y_train.rows(), "sigmoid",ops[3], "he",1);

        // Use MPI to pass layer to its respective process
        // Example: layer(1) should be passed to rank(1)
        // General rank(0) is used for combining results

        // We use L+1 because of passing useless layers[0] to rank(0)
        std::vector<FCLayer> layers(L+1);
        layers[0] = l1;
        layers[1] = l1;
        layers[2] = l2;
        layers[3] = l3;
        layers[4] = l4;
        mpi::scatter(world, layers, layer, 0);
        mpi::broadcast(world, epochs, 0);
        mpi::request reqs[3];
        reqs[0] = world.isend(1, 3, X_train);
        reqs[1] = world.isend(world.size()-1, 2, Y_train);
        reqs[2] = world.isend(world.size()-1, 4, loss_type);
        mpi::wait_all(reqs, reqs+3);
        std::cout << "DONE SENDING INITIAL DATA" << std::endl;



    }else{
        mpi::scatter(world, layers, layer, 0);
        mpi::broadcast(world, epochs, 0);
        md AL;
        md X;
        md Y;
        LossWrapper loss;
        std::string loss_type;

        if (slaves.rank() == 0){
            mpi::request reqs[1];
            reqs[0] = world.irecv(0, 3, X);
            mpi::wait_all(reqs, reqs+1);
        }
        if (slaves.rank() == slaves.size() - 1){
            mpi::request reqs[2];
            reqs[0] = world.irecv(0, 2, Y);
            reqs[1] = world.irecv(0, 4, loss_type);
            mpi::wait_all(reqs, reqs+2);

        }
        if (slaves.rank() == slaves.size() - 1){
            loss = LossWrapper(loss_type);
        }

        for (int e=0; e < 2*epochs; ++e){
            slaves.barrier();
            std::cout << e << std::endl;

            if (stage_identifier_forward){
                if (slaves.rank() != 0){
                    slaves.recv(slaves.rank()-1, 1, X);
                }
                std::cerr << "s1" << std::endl;
                if (slaves.rank() == 1){
                    std::cerr << "WAS" << std::endl;
                    std::cerr << X << std::endl;
                }
                AL = layer.forward(X);
                std::cerr << "s2" << std::endl;

                if (slaves.rank() != slaves.size() - 1){
                    slaves.send(slaves.rank()+1, 1, AL);
                }
                stage_identifier_forward = !stage_identifier_forward;

            }else{
                md dA;
                if (slaves.rank() != slaves.size() - 1){
                    slaves.recv(slaves.rank()+1, 2, dA);

                }else{
                    std::cerr << "h1" << std::endl;
                    dA = loss.get_loss_backward(AL, Y);
                    std::cerr << "h2" << std::endl;

                }
                std::cerr << "h3" << std::endl;
                dA = layer.backward(dA);
                std::cerr << "h4" << std::endl;

                layer.update_parameters(1);
                std::cerr << "h5" << std::endl;


                if (slaves.rank() != 0){
                    slaves.send(slaves.rank()-1, 2, dA);
                }
                stage_identifier_forward = !stage_identifier_forward;


            }

        }

    }




//    auto op = new SGD(0.01);
//    model->fit_data_parallel(X_train, Y_train, 10, true);




//    auto model = new Model{};
//    model->add(l1);
//    model->add(l2);
//    model->add(l3);
//    model->add(l4);
//    auto op = new Adam(0.01);
//    auto loss = new BinaryCrossEntropy();
//    model->compile(loss);
//    model->fit(X_train, Y_train, 10, true, 24);
//    auto res = model->evaluate(X_test, Y_test);
//    std::cout << "Test accuracy = " << std::to_string(res) << std::endl;

}





int main(int argc, char **argv) {

    test_fc_layer_mpi(argc, argv);

}

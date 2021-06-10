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

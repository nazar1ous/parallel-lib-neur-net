
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

    std::fstream in_file("/home/nazariikuspys/UCU/AKS/parallel-lib-neur-net/tests/dataset.txt");
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

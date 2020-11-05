#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layer.hpp"
#include <armadillo>

using namespace std;
using namespace arma;

class FFNN {
private:
    friend class Layer;

    vector<Layer> layers_;

    int hidden_layers_, features_, nodes_, num_outputs_, num_points_, num_layers_;
    int epochs_, batch_sz_;
    double eta_;
    mat X_train_, y_train_;
    mat X_test_, y_test_;

    void feed_forward(vec x);
    void backward_pass(vec x, vec y);
    void add_gradients(int l);
    void update_parameters();

    //Pointers to member functions
    vec (FFNN::*hidden_act)(vec z);
    vec (FFNN::*top_layer_act)(vec a);
    vec (FFNN::*hidden_act_derivative)(vec z);


    //Various hidden layer activation functions
    vec sigmoid(vec z);
    vec sigmoid_derivative(vec z);

    vec relu(vec z);
    vec relu_derivative(vec z);

    //Top layer activation functions
    vec softmax(vec a);




public:
    FFNN(int hidden_layers, int features, int nodes, int outputs);
    void init_data(mat X_train, mat y_train, int num_points);
    void fit(int epochs, int batch_sz, double eta);
    void evaluate(mat X_test, mat y_test, int num_test);
};

#endif

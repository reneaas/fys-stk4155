#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layer.hpp"
#include <armadillo>

using namespace std;
using namespace arma;



/*
The class FFNN is a feed forward neural network. There are several constructors depending on the problem type:


*/

class FFNN {
private:
    friend class Layer;

    vector<Layer> layers_;

    int hidden_layers_, features_, nodes_, num_outputs_, num_points_, num_layers_;
    int epochs_, batch_sz_;
    double eta_;
    mat X_train_, y_train_;
    mat X_test_, y_test_;
    int num_test_;
    double lamb_, gamma_;


    //These methods make up the backpropagation algorithm
    void feed_forward(vec x);
    void backward_pass(vec x, vec y);
    void add_gradients(int l);

    //Pointers to member functions
    vec (FFNN::*hidden_act)(vec z);
    vec (FFNN::*top_layer_act)(vec a);
    vec (FFNN::*hidden_act_derivative)(vec z);
    double (FFNN::*compute_metric)();
    void (FFNN::*update_parameters)();


    //Various hidden layer activation functions
    vec sigmoid(vec z);
    vec sigmoid_derivative(vec z);

    vec relu(vec z);
    vec relu_derivative(vec z);

    vec leaky_relu(vec z);
    vec leaky_relu_derivative(vec z);

    //Top layer activation functions
    vec softmax(vec a);
    vec linear(vec z);
    vec binary_classifier(vec z);

    //Various metrics
    double compute_accuracy();
    double compute_r2();
    double compute_mse();


    //Various update rules
    void update();
    void update_l2();
    void update_l2_momentum();


public:
    //Constructors
    FFNN(int hidden_layers, int features, int nodes, int outputs, string model_type);
    FFNN(int hidden_layers, int features, int nodes, int outputs, string model_type, double lamb);
    FFNN(int hidden_layers, int features, int nodes, int outputs, string model_type, double lamb, double gamma, string hidden_activation);

    //Member functions
    void init_data(mat X_train, mat y_train, int num_points); //Initializes training data.
    void fit(int epochs, int batch_sz, double eta); //Fits the model
    double evaluate(mat X_test, mat y_test, int num_test); //Evaluates the model according to the appropriate performance metric.
};

#endif

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>

using namespace std;


class FFNN {
private:

    //To store dataset
    double *X_data_, *y_data_;
    //Parameters of the network
    double *weights_, *biases_, *activations_, *z_;
    double *dw_, *db_, *error_;
    double *y_;

    int nodes_, layers_, features_, num_outputs_, epochs_, batch_size_, num_points_;
    double eta_, lambda_;
    int *num_rows_, *num_cols_, *r_w_, *r_b_, *r_a_;

    //Backprop algo functions
    void feed_forward();
    void backward_pass();
    void update();
    void update_l2();

    //Pointer to member functions.
    void (FFNN::*top_layer_act)();
    double (FFNN::*hidden_act)(double z);
    double (FFNN::*hidden_act_derivative)(double z);
    void (FFNN::*update_parameters)();

    //Top layer activation functions
    void predict_linear();
    void predict_softmax();

    //Various activation functions
    double sigmoid(double z);
    double sigmoid_derivative(double z);

    double relu(double z);
    double relu_derivative(double z);

    double leaky_relu(double z);
    double leaky_relu_derivative(double z);


public:
    //Constructors
    FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type);
    FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type, string hidden_act);
    FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type, string hidden_activation, double lambda);
    //FFNN(int test);

    //void test_func();

    ~FFNN();
    void create_model_arch();
    void init_parameters(); //Sets up initial weights and biases.
    void init_data(double *X_data, double *y_data, int num_points);
    void fit();
    void evaluate(double *X_test, double *y_test, int num_test);
};


#endif

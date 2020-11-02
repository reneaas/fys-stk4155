#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <cmath>
#include <random>
#include <omp.h>

using namespace std;


class FFNN {
private:

    //To store dataset
    double *X_data_, *y_data_;

    //Parameters of the network
    double *weights_, *biases_, *activations_;
    double *dw_, *db_, *error_;

    double *y_;

    int nodes_, layers_, features_, num_outputs_, epochs_, batch_size_;
    double eta_;
    int *num_rows_, *num_cols_, *r_w_, *r_b_, *r_a_;

    int num_points_;

    double sigmoid(double z);
    double softmax();
    void feed_forward();
    void backward_pass();
    void update_parameters();
    //void clean_up_gradients();

public:
    FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features);
    ~FFNN();
    void init_data(double *X_data, double *y_data, int num_points);
    void fit();
    void predict(double *X_test, double *y_test, int num_test);
};


#endif

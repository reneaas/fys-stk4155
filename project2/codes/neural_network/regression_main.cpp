#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <fstream>

using namespace std;
using namespace arma;

void read_franke(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int deg, int num_train, int num_val, int num_test);
void test_franke(int hidden_layers, int nodes, double lamb, double gamme, int epochs, int batch_sz, double eta, string hidden_act, int deg);

int main(int argc, char const *argv[]) {

    int hidden_layers = 1;
    int nodes = 50;
    double lamb = 1e-5;
    double gamma = 0.9;
    int epochs = 100;
    int batch_sz = 10;
    double eta = 0.01;
    string hidden_act = "relu";
    int deg = 5;
    test_franke(hidden_layers, nodes, lamb, gamma, epochs, batch_sz, eta, hidden_act, deg);

    return 0;
}

void test_franke(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string hidden_act, int deg){
    mat X_train, y_train;
    mat X_val, y_val;
    mat X_test, y_test;

    int num_points = 20000;
    int num_train = 0.9*0.95*num_points;
    int num_val = 0.9*0.05*num_points;
    int num_test = 0.1*num_points;

    read_franke(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test, deg, num_train, num_val, num_test);

    int features = 2;
    string model_type = "regression";
    int num_outputs = 1;

    //Here we show how to use the neural net with several hidden layers with a variable number of nodes in the hidden layers.
    FFNN my_network(features, num_outputs, model_type, lamb, gamma, hidden_act);
    my_network.add_layer(nodes, features); //Add the first hidden layer connected to the input x.
    my_network.add_layer(2*nodes, nodes); //Add a hidden layer
    my_network.add_layer(5*nodes, 2*nodes); //Add another hidden layer
    my_network.add_layer(3*nodes, 5*nodes); //Add yet another hidden layer.
    my_network.add_layer(num_outputs, 3*nodes); //add top layer that produces the final prediction
    my_network.init_data(X_train, y_train); //Feed the training data to the model
    my_network.fit(epochs, batch_sz, eta); //Fit the model to the training data.
    double r2_val = my_network.evaluate(X_val, y_val); //Evaluate the model on the validation set
    double r2_test = my_network.evaluate(X_test, y_test); //Evaluate the model on the test set.

    cout << "validation R2 = " << r2_val << endl;
    cout << "test R2 = " << r2_test << endl;


    //Here we show a different version where every hidden layer has the same number of hidden neurons.
    //It calls a different constructor that sets up the model for us with no fuzz.
    FFNN my_network2(hidden_layers, features, nodes, num_outputs, model_type, lamb, gamma, hidden_act); //Create the neural net with all its layers.
    my_network2.init_data(X_train, y_train); //Feed the training data to the model
    my_network2.fit(epochs, batch_sz, eta); //Fit the model to the training data.
    r2_val = my_network2.evaluate(X_val, y_val); //Evalute the model on the validation set
    r2_test = my_network2.evaluate(X_test, y_test); //Evalute the model on the test set.

    cout << "validation R2 = " << r2_val << endl;
    cout << "test R2 = " << r2_test << endl;


}


/*
Reads and creates data matrices of Franke's function for the training-, validation- and test set.
*/
void read_franke(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int deg, int num_train, int num_val, int num_test)
{
    int n = 20000;
    vec x = vec(n);
    vec y = vec(n);
    vec z = vec(n);
    x.load("../datasets/frankefunction_x_20000_sigma_0.1.bin");
    y.load("../datasets/frankefunction_y_20000_sigma_0.1.bin");
    z.load("../datasets/frankefunction_z_20000_sigma_0.1.bin");


    int features = 2;

    (*X_train) = mat(features, num_train);
    (*y_train) = mat(1, num_train);

    (*X_val) = mat(features, num_val);
    (*y_val) = mat(1, num_val);

    (*X_test) = mat(features, num_test);
    (*y_test) = mat(1, num_test);

    for (int i = 0; i < num_train; i++){
        (*y_train)(0, i) = z(i);
        (*X_train)(0, i) = x(i);
        (*X_train)(1,i) = y(i);
    }

    for (int i = 0; i < num_val; i++){
        (*y_val)(0, i) = z(num_train + i);
        (*X_val)(0, i) = x(num_train + i);
        (*X_val)(1, i) = y(num_train + i);
    }

    for (int i = 0; i < num_test; i++){
        (*y_test)(0, i) = z(num_train + num_val + i);
        (*X_test)(0, i) = x(num_train + num_val + i);
        (*X_test)(1, i) = y(num_train + num_val + i);
    }
}

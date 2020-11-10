#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <fstream>

using namespace std;
using namespace arma;



void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);
void test_mnist(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string hidden_act);

int main(int argc, char const *argv[]) {

    int hidden_layers = 1;
    int nodes = 10;
    double lamb = 1e-5;
    double gamma = 0.5;
    int epochs = 10;
    int batch_sz = 10;
    double eta = 0.01;
    string hidden_act = "relu";
    test_mnist(hidden_layers, nodes, lamb, gamma, epochs, batch_sz, eta, hidden_act);
    return 0;
}

void test_mnist(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string hidden_act)
{
    int features = 28*28;
    int num_outputs = 10;

    int num_train = 0.95*60000;
    mat X_train = mat(features, num_train);
    mat y_train = mat(num_outputs, num_train);

    int num_val = 0.05*60000;
    mat X_val = mat(features, num_val);
    mat y_val = mat(num_outputs, num_val);

    int num_test = 10000;
    mat X_test = mat(features, num_test);
    mat y_test = mat(num_outputs, num_test);


    read_mnist(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test);
    string model_type = "classification";

    FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type, lamb, gamma, hidden_act);
    my_network.init_data(X_train, y_train, num_train);
    my_network.fit(epochs, batch_sz, eta);

    double accuracy_val = my_network.evaluate(X_val, y_val, num_val);
    double accuracy_test = my_network.evaluate(X_test, y_test, num_test);

    cout << "validation accuracy = " << accuracy_val << endl;
    cout << "test accuracy = " << accuracy_test << endl;

}

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test){

    (*X_train).load("../datasets/mnist_X_train.bin");
    (*y_train).load("../datasets/mnist_y_train.bin");

    (*X_val).load("../datasets/mnist_X_val.bin");
    (*y_val).load("../datasets/mnist_y_val.bin");


    (*X_test).load("../datasets/mnist_X_test.bin");
    (*y_test).load("../datasets/mnist_y_test.bin");
}

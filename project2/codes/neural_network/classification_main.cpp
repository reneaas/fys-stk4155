#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <fstream>

using namespace std;
using namespace arma;



void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);
void test_mnist(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string outfilename, string hidden_act);

int main(int argc, char const *argv[]) {

    int hidden_layers = atoi(argv[1]);
    int nodes = atoi(argv[2]);
    double lamb = atof(argv[3]);
    double gamma = atof(argv[4]);
    int epochs = atoi(argv[5]);
    int batch_sz = atoi(argv[6]);
    double eta = atof(argv[7]);
    string outfilename = argv[8];
    string hidden_act = argv[9];

    test_mnist(hidden_layers, nodes, lamb, gamma, epochs, batch_sz, eta, outfilename, hidden_act);


    return 0;
}

void test_mnist(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string outfilename, string hidden_act)
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

    //FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type);
    FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type, lamb, gamma, hidden_act);
    my_network.init_data(X_train, y_train, num_train);

    clock_t start = clock();
    my_network.fit(epochs, batch_sz, eta);
    clock_t end = clock();
    double timeused = (double) (end-start)/CLOCKS_PER_SEC;
    cout << "timeused = " << timeused << endl;

    double accuracy_val = my_network.evaluate(X_val, y_val, num_val);
    double accuracy_test = my_network.evaluate(X_test, y_test, num_test);

    ofstream ofile;
    ofile.open(outfilename);
    ofile << accuracy_val << " " << accuracy_test << endl;
    ofile.close();

}

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test){

    (*X_train).load("./data_files/mnist_X_train.bin");
    (*y_train).load("./data_files/mnist_y_train.bin");

    (*X_val).load("./data_files/mnist_X_val.bin");
    (*y_val).load("./data_files/mnist_y_val.bin");


    (*X_test).load("./data_files/mnist_X_test.bin");
    (*y_test).load("./data_files/mnist_y_test.bin");
}

#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <fstream>

using namespace std;
using namespace arma;

void read_franke(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int deg, int num_train, int num_val, int num_test);
void test_franke(int hidden_layers, int nodes, double lamb, double gamme, int epochs, int batch_sz, double eta, string outfilename, string hidden_act, int deg);

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
    int deg = atoi(argv[10]);


    test_franke(hidden_layers, nodes, lamb, gamma, epochs, batch_sz, eta, outfilename, hidden_act, deg);

    return 0;
}

void test_franke(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string outfilename, string hidden_act, int deg){
    mat X_train, y_train;
    mat X_val, y_val;
    mat X_test, y_test;

    int num_points = 20000;
    int num_train = 0.9*0.95*num_points;
    int num_val = 0.9*0.05*num_points;
    int num_test = 0.1*num_points;

    read_franke(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test, deg, num_train, num_val, num_test);

    int features = (int) (deg+1)*(deg+2)/2;
    string model_type = "regression";
    int num_outputs = 1;


    //FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type);
    FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type, lamb, gamma, hidden_act);

    my_network.init_data(X_train, y_train, num_train);

    my_network.fit(epochs, batch_sz, eta);
    double r2_val = my_network.evaluate(X_val, y_val, num_val);
    double r2_test = my_network.evaluate(X_test, y_test, num_test);

    ofstream ofile;
    ofile.open(outfilename);
    ofile << r2_val << " " << r2_test << endl;
    ofile.close();
}

void read_franke(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int deg, int num_train, int num_val, int num_test)
{
    int n = 20000;
    vec x = vec(n);
    vec y = vec(n);
    vec z = vec(n);
    x.load("../datasets/frankefunction_x_20000_sigma_0.1.bin");
    y.load("../datasets/frankefunction_y_20000_sigma_0.1.bin");
    z.load("../datasets/frankefunction_z_20000_sigma_0.1.bin");


    int features = (deg+1)*(deg+2)/2;

    (*X_train) = mat(features, num_train);
    (*y_train) = mat(1, num_train);

    (*X_val) = mat(features, num_val);
    (*y_val) = mat(1, num_val);

    (*X_test) = mat(features, num_test);
    (*y_test) = mat(1, num_test);

    int q;
    for (int i = 0; i < num_train; i++){
        (*y_train)(0, i) = z(i);
        (*X_train)(0, i) = 1.;
        for (int j = 1; j < deg+1; j++){
            q = (int) (j*(j+1)/2);
            for (int k = 0; k < j+1; k++){
                (*X_train)(q+k, i) = pow(x(i), j-k)*pow(y(i), k);
            }
        }
    }

    for (int i = 0; i < num_val; i++){
        (*y_val)(0, i) = z(num_train + i);
        (*X_val)(0, i) = 1.;
        for (int j = 1; j < deg+1; j++){
            q = (int) (j*(j+1)/2);
            for (int k = 0; k < j+1; k++){
                (*X_val)(q+k, i) = pow(x(num_train + i), j-k)*pow(y(num_train + i), k);
            }
        }
    }

    for (int i = 0; i < num_test; i++){
        (*y_test)(0, i) = z(num_train + num_val + i);
        (*X_test)(0, i) = 1.;
        for (int j = 1; j < deg+1; j++){
            q = (int) (j*(j+1)/2);
            for (int k = 0; k < j+1; k++){
                (*X_test)(q+k, i) = pow(x(num_train + num_val +i), j-k)*pow(y(num_train  + num_val + i), k);
            }
        }
    }
}

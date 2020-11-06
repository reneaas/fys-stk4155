#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);
void test_mnist();

void read_franke(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int deg, int num_train, int num_val, int num_test);
void test_franke();


int main(int argc, char const *argv[]) {
    test_franke();
    //test_mnist();
    return 0;
}


void test_mnist()
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

    int hidden_layers = 1;
    int nodes = 30;
    int epochs = 10;
    int batch_sz = 10;
    double eta = 0.5;
    string model_type = "classification";

    FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type);
    my_network.init_data(X_train, y_train, num_train);

    clock_t start = clock();
    my_network.fit(epochs, batch_sz, eta);
    clock_t end = clock();
    double timeused = (double) (end-start)/CLOCKS_PER_SEC;
    cout << "timeused = " << timeused << endl;

    my_network.evaluate(X_val, y_val, num_val);
    my_network.evaluate(X_test, y_test, num_test);
}

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test){

    (*X_train).load("./data_files/mnist_X_train.bin");
    (*y_train).load("./data_files/mnist_y_train.bin");

    (*X_val).load("./data_files/mnist_X_val.bin");
    (*y_val).load("./data_files/mnist_y_val.bin");


    (*X_test).load("./data_files/mnist_X_test.bin");
    (*y_test).load("./data_files/mnist_y_test.bin");
}


void test_franke(){
    mat X_train, y_train;
    mat X_val, y_val;
    mat X_test, y_test;

    int deg = 8;
    int num_points = 20000;
    int num_train = 0.9*0.95*num_points;
    int num_val = 0.9*0.05*num_points;
    int num_test = 0.1*num_points;

    read_franke(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test, deg, num_train, num_val, num_test);


    int hidden_layers = 1;
    int features = (int) (deg+1)*(deg+2)/2;
    int num_outputs = 1;
    int nodes = 30;
    string model_type = "regression";

    int epochs = 30;
    int batch_sz = 10;
    double eta = 0.3;

    double lamb = 1e-5;


    //FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type);
    FFNN my_network(hidden_layers, features, nodes, num_outputs, model_type, lamb);

    my_network.init_data(X_train, y_train, num_train);

    my_network.fit(epochs, batch_sz, eta);
    my_network.evaluate(X_val, y_val, num_val);



}

void read_franke(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int deg, int num_train, int num_val, int num_test)
{
    int n = 20000;
    vec x = vec(n);
    vec y = vec(n);
    vec z = vec(n);
    x.load("./data_files/frankfunction_x_20000_sigma_0.1.bin");
    y.load("./data_files/frankefunction_y_20000_sigma_0.1.bin");
    z.load("./data_files/frankefunction_z_20000_sigma_0.1.bin");


    int features = (deg+1)*(deg+2)/2;

    (*X_train) = mat(num_train, features);
    (*y_train) = mat(num_train, 1);

    (*X_val) = mat(num_val, features);
    (*y_val) = mat(num_val, 1);

    (*X_test) = mat(num_test, features);
    (*y_test) = mat(num_test, 1);

    int col_idx;
    int max_deg;
    int q;

    for (int i = 0; i < num_train; i++){
        (*y_train)(i, 0) = z(i);
        (*X_train)(i, 0) = 1.;
        for (int j = 1; j < deg+1; j++){
            q = (int) (j*(j+1)/2);
            for (int k = 0; k < j+1; k++){
                (*X_train)(i, q+k) = pow(x(i), j-k)*pow(y(i), k);
            }
        }
    }

    for (int i = 0; i < num_val; i++){
        (*y_val)(i, 0) = z(num_train + i);
        (*X_val)(i, 0) = 1.;
        for (int j = 1; j < deg+1; j++){
            q = (int) (j*(j+1)/2);
            for (int k = 0; k < j+1; k++){
                (*X_val)(i, q+k) = pow(x(num_train + i), j-k)*pow(y(num_train + i), k);
            }
        }
    }

    for (int i = 0; i < num_test; i++){
        (*y_test)(i, 0) = z(i);
        (*X_test)(i, 0) = 1.;
        for (int j = 1; j < deg+1; j++){
            q = (int) (j*(j+1)/2);
            for (int k = 0; k < j+1; k++){
                (*X_test)(i, q+k) = pow(x(num_train + num_val +i), j-k)*pow(y(num_train  + num_val + i), k);
            }
        }
    }

    (*X_train) = (*X_train).t();
    (*y_train) = (*y_train).t();

    (*X_val) = (*X_val).t();
    (*y_val) = (*y_val).t();

    (*X_test) = (*X_test).t();
    (*y_test) = (*y_test).t();

}

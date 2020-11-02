#include "neural_network.hpp"
#include <time.h>
#include <cstdio>

void read_mnist(char* filename_X, char* filename_Y, double *X_train, double *y_train, int N_train, int features, int n_classes);
void test_mnist();
void test_algo();

int main(int argc, char const *argv[]) {

    test_mnist();
    //test_algo();

    return 0;
}

void read_mnist(char* filename_X, char* filename_Y, double *X, double *y, int N, int features, int n_classes){
    FILE *fp_X = fopen(filename_X, "r");
    FILE *fp_Y = fopen(filename_Y, "r");
    for (int i = 0; i < N; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &X[i*features + j]);
        }
        for (int k = 0; k < n_classes; k++){
            fscanf(fp_Y, "%lf", &y[i*n_classes + k]);
        }
    }
    fclose(fp_X);
    fclose(fp_Y);
}


void test_mnist(){
    string problem_type = "classification";
    string hidden_act = "sigmoid";
    int hidden_layers = 1;
    int nodes = 30;
    int N_outputs = 10;
    int epochs = 10;
    int batch_size = 10;
    double eta = 0.1;
    double lambda = 1e-5;
    double gamma = 0.7;
    int features = 28*28;

    int N_train = 60000;
    int n_classes = 10;
    double *X_train = new double[N_train*features];
    double *y_train = new double[N_train*n_classes];
    char* filename_X = "./data_files/mnist_training_X.txt";
    char* filename_Y = "./data_files/mnist_training_Y.txt";
    read_mnist(filename_X, filename_Y, X_train, y_train, N_train, features, n_classes);

    FFNN my_network(hidden_layers, nodes, N_outputs, epochs, batch_size, eta, features, problem_type, hidden_act);
    my_network.init_data(X_train, y_train, N_train);
    clock_t start = clock();
    my_network.fit();
    clock_t end = clock();
    double timeused = (double)(end-start)/CLOCKS_PER_SEC;
    cout << "CPU time = " << timeused << " seconds " << endl;



    int N_test = 10000;
    double *X_test = new double[N_test*features];
    double *y_test = new double[N_test*n_classes];
    filename_X = "./data_files/mnist_test_X.txt";
    filename_Y = "./data_files/mnist_test_Y.txt";
    read_mnist(filename_X, filename_Y, X_test, y_test, N_test, features, n_classes);
    my_network.evaluate(X_test, y_test, N_test);
}

/*
void test_algo(){
    int features = 2;

    //Create data point
    double *x = new double[features];
    x[0] = 1.;
    x[1] = 2.;

    int classes = 1;
    double *y = new double[classes];
    y[0] = 1.; //Response variable

    int test = 1;
    FFNN my_network(test);
    my_network.init_data(x, y, 1);
    my_network.test_func();
}
*/

#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>

void read_mnist(mat *X_train, mat *y_train, mat *X_test, mat *y_test, int num_train, int num_test);


int main(int argc, char const *argv[]) {

    /*
    int features = 2;
    int num_outputs = 2;
    int nodes = 3;
    int num_train = 1;

    int hidden_layers = 1;


    mat X_train = mat(features, num_train);
    mat y_train = mat(num_outputs, num_train);

    X_train(0, 0) = 1.;
    X_train(1, 0) = -2.;

    y_train(0, 0) = 0.;
    y_train(1, 0) = 1.;




    //mat X_train = randu<mat>(features, num_train);
    //mat y_train = randu<mat>(num_outputs, num_train);

    //X_train.print("X train = ");
    //y_train.print("y train = ");

    FFNN my_network(hidden_layers, features, nodes, num_outputs);
    my_network.init_data(X_train, y_train, num_train);
    my_network.test_init();

    int epochs = 1;
    int batch_sz = 1;
    double eta = 0.1;
    my_network.fit(epochs, batch_sz, eta);
    */


    /*
    vec v = linspace(-1, 1, 10);
    //uvec idx = find(u <= 0);
    vec u = clamp(v, 0, v.max());
    v.print("v = ");
    u.print("u = ");
    */






    int features = 28*28;
    int num_outputs = 10;

    int num_train = 60000;
    //mat X_train = mat(features, num_train);
    //mat y_train = mat(num_outputs, num_train);
    mat X_train = mat(num_train, features);
    mat y_train = mat(num_train, num_outputs);

    int num_test = 10000;
    //mat X_test = mat(features, num_test);
    //mat y_test = mat(num_outputs, num_test);
    mat X_test = mat(num_test, features);
    mat y_test = mat(num_test, num_outputs);

    read_mnist(&X_train, &y_train, &X_test, &y_test, num_train, num_test);

    int hidden_layers = 1;
    int nodes = 30;
    int epochs = 30;
    int batch_sz = 10;
    double eta = 0.5;

    FFNN my_network(hidden_layers, features, nodes, num_outputs);
    my_network.init_data(X_train, y_train, num_train);

    clock_t start = clock();
    my_network.fit(epochs, batch_sz, eta);
    clock_t end = clock();
    double timeused = (double) (end-start)/CLOCKS_PER_SEC;
    cout << "timeused = " << timeused << endl;


    my_network.evaluate(X_test, y_test, num_test);
    






    return 0;
}


void read_mnist(mat *X_train, mat *y_train, mat *X_test, mat *y_test, int num_train, int num_test){
    int features = 28*28;
    int num_outputs = 10;

    char* infilename_X = "./data_files/mnist_training_X.txt";
    char* infilename_y = "./data_files/mnist_training_Y.txt";
    FILE *fp_X = fopen(infilename_X, "r");
    FILE *fp_y = fopen(infilename_y, "r");
    for (int i = 0; i < num_train; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &(*X_train)(i, j));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &(*y_train)(i, j));
        }
    }
    fclose(fp_X);
    fclose(fp_y);

    (*X_train) = (*X_train).t();
    (*y_train) = (*y_train).t();

    infilename_X = "./data_files/mnist_test_X.txt";
    infilename_y = "./data_files/mnist_test_Y.txt";
    fp_X = fopen(infilename_X, "r");
    fp_y = fopen(infilename_y, "r");
    for (int i = 0; i < num_test; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &(*X_test)(i, j));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &(*y_test)(i, j));
        }
    }
    fclose(fp_X);
    fclose(fp_y);

    (*X_test) = (*X_test).t();
    (*y_test) = (*y_test).t();

}

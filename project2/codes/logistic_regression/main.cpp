#include "LogReg.hpp"

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);

void SGD(int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test);

void SGD_momentum(double gamma, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test);

void ADAM(double beta1, double beta2, double epsilon, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test);


int main(int argc, char const *argv[]) {

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

    //Read minst data to matrices
    read_mnist(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test);

    int epochs = 10;
    int batch_sz = 500;
    double eta = 0.1;
    int classes = 10;
    double Lambda = 1e-4;


    //Train the model using SGD without momentum and make predictions on unseen data
    cout << " " << endl;
    cout << "Training model using SGD:" << endl;
    SGD(classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features, X_val, y_val, num_val);


    //Train the model using SGD with momentum and make predictions on unseen data
    cout << " " << endl;
    cout << "Training model using SGD w/momentum:" << endl;
    double gamma = 1e-7;
    SGD_momentum(gamma, classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features, X_val, y_val, num_val);

    //Train the model using Adam and make predictions on unseen data
    cout << " " << endl;
    cout << "Training model using ADAM:" << endl;
    double beta1 = 0.99;
    double beta2 = 0.99;
    double epsilon = 1e-8;
    ADAM(beta1, beta2, epsilon, classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features, X_test, y_test, num_test);

    return 0;
}

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test){

    (*X_train).load("../datasets/mnist_X_train.bin");
    (*y_train).load("../datasets/mnist_y_train.bin");

    (*X_val).load("../datasets/mnist_X_val.bin");
    (*y_val).load("../datasets/mnist_y_val.bin");


    (*X_test).load("../datasets/mnist_X_test.bin");
    (*y_test).load("../datasets/mnist_y_test.bin");
}

void SGD(int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test){
    LogReg my_model_SGD(classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features);
    my_model_SGD.fit();
    double accuracy_SGD = my_model_SGD.compute_accuracy(X_test, y_test, num_test);
}

void SGD_momentum(double gamma, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test){
    LogReg my_model_SGD_momentum(gamma, classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features);
    my_model_SGD_momentum.fit();
    double accuracy_SGD_momentum = my_model_SGD_momentum.compute_accuracy(X_test, y_test, num_test);
}

void ADAM(double beta1, double beta2, double epsilon, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test){
    LogReg my_model_ADAM(beta1, beta2, epsilon, classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features);
    my_model_ADAM.fit();
    double accuracy_ADAM = my_model_ADAM.compute_accuracy(X_test, y_test, num_test);
}

#include "LogReg.hpp"

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);

void read_mnist_to_bin(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int num_train, int num_val, int num_test);

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

    //Read minst data in to binary files for easy access
    //read_mnist_to_bin(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test, num_train, num_val, num_test);


    //Read minst data to matrices
    read_mnist(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test);

    int epochs = 10;
    int batch_sz = 10;
    double eta = 0.5;
    string optimizer = "ADAM";
    int classes = 10;
    double Lambda = 1e-2;
    double gamma = 0.1;

    LogReg my_model(classes, X_train, y_train, eta, gamma, Lambda, epochs, batch_sz, optimizer, num_train, features);
    my_model.fit();


    return 0;
}

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test){

    (*X_train).load("mnist_X_train.bin");
    (*y_train).load("mnist_y_train.bin");

    (*X_val).load("mnist_X_val.bin");
    (*y_val).load("mnist_y_val.bin");


    (*X_test).load("mnist_X_test.bin");
    (*y_test).load("mnist_y_test.bin");
}


void read_mnist_to_bin(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int num_train, int num_val, int num_test){
    int features = 28*28;
    int num_outputs = 10;


    char* infilename_X = "mnist_training_X.txt";
    char* infilename_y = "mnist_training_Y.txt";
    FILE *fp_X = fopen(infilename_X, "r");
    FILE *fp_y = fopen(infilename_y, "r");
    for (int i = 0; i < num_train; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &(*X_train)(j, i));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &(*y_train)(j, i));
        }
    }

    for (int i = 0; i < num_val; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &(*X_val)(j, i));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &(*y_val)(j, i));
        }
    }

    fclose(fp_X);
    fclose(fp_y);

    (*X_train).save("mnist_X_train.bin");
    (*y_train).save("mnist_y_train.bin");

    (*X_val).save("mnist_X_val.bin");
    (*y_val).save("mnist_y_val.bin");




    infilename_X = "mnist_test_X.txt";
    infilename_y = "mnist_test_Y.txt";
    fp_X = fopen(infilename_X, "r");
    fp_y = fopen(infilename_y, "r");
    for (int i = 0; i < num_test; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &(*X_test)(j, i));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &(*y_test)(j, i));
        }
    }
    fclose(fp_X);
    fclose(fp_y);

    (*X_test).save("mnist_X_test.bin");
    (*y_test).save("mnist_y_test.bin");

}

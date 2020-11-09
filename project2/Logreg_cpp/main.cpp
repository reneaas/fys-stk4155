#include "LogReg.hpp"


void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);

void SGD(int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test);

void SGD_momentum(double gamma, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test);

void ADAM(double beta1, double beta2, double epsilon, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features, mat X_test, mat y_test, int num_test);

void SGD_hyperparams(int classes, mat X_train, mat y_train, int num_train, int features, mat X_test, mat y_test, int num_test, mat X_val, mat y_val, int num_val);

void SGD_mom_hyperparams(int classes, mat X_train, mat y_train, int num_train, int features, mat X_test, mat y_test, int num_test, mat X_val, mat y_val, int num_val);

void ADAM_hyperparams(int classes, mat X_train, mat y_train, int num_train, int features, mat X_test, mat y_test, int num_test, mat X_val, mat y_val, int num_val);

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

    //int epochs = 20;
    //int batch_sz = 300;
    //double eta = 0.1;
    int classes = 10;
    //double Lambda = 1e-8;

    //Optimizer = SGD:
    /*
    SGD(classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features, X_val, y_val, num_val);
    SGD_hyperparams(classes, X_train, y_train, num_train, features, X_test, y_test, num_test, X_val, y_val, num_val);

    */



    //Optimizer = SGD with momentum:
    /*
    double gamma = 0.5;
    SGD_momentum(gamma, classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features, X_val, y_val, num_val);

    SGD_mom_hyperparams(classes, X_train, y_train, num_train, features, X_test, y_test, num_test, X_val, y_val, num_val);
    SGD_mom_hyperparams(classes, X_train, y_train, num_train, features, X_test, y_test, num_test, X_val, y_val, num_val);
    */



    //Optimizer = ADAM;
    /*
    double beta1 = 0.99;
    double beta2 = 0.90;
    double epsilon = 1e-8;
    ADAM(beta1, beta2, epsilon, classes, X_train, y_train, eta, Lambda, epochs, batch_sz, num_train, features, X_test, y_test, num_test);
    ADAM_hyperparams(classes, X_train, y_train, num_train, features, X_test, y_test, num_test, X_val, y_val, num_val);
    ADAM_hyperparams(classes, X_train, y_train, num_train, features, X_test, y_test, num_test, X_val, y_val, num_val);
    */








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
    double accuracy_ADAM= my_model_ADAM.compute_accuracy(X_test, y_test, num_test);
}

void SGD_hyperparams(int classes, mat X_train, mat y_train, int num_train, int features, mat X_test, mat y_test, int num_test, mat X_val, mat y_val, int num_val){

    double Lambda = 0.;
    double accuracy_t, accuracy_v;


    int epochs_list[6] = {1, 10, 20, 50, 100, 1000};
    int batch_list[5] = {10, 100, 200, 300, 500};
    double etaa = 0.1;

    ofstream ofile;
    ofile.open("SGD_LogReg_final.txt");
    ofile << "Acc_val " << "Acc_test " << "epochs " << "batch_sz " << endl;
    for (int ep : epochs_list){
        for(int b : batch_list){
            LogReg my_model(classes, X_train, y_train, etaa, Lambda, ep, b, num_train, features);
            my_model.fit();
            accuracy_v = my_model.compute_accuracy(X_val, y_val, num_val);
            accuracy_t = my_model.compute_accuracy(X_test, y_test, num_test);

            ofile << accuracy_v << " " << accuracy_t << " " << ep << " " << b <<endl;



        }
    }
    ofile.close();
}

void SGD_mom_hyperparams(int classes, mat X_train, mat y_train, int num_train, int features, mat X_test, mat y_test, int num_test, mat X_val, mat y_val, int num_val){

    double accuracy_t, accuracy_v;

    int epochss = 100;
    int batch_szz = 500;
    double etaa = 0.1;
    double gamma_list[9] = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0};
    double Lambda_list[9] = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0};

    ofstream ofile;
    ofile.open("SGD_mom_LogReg_final.txt");
    ofile << "Acc_val " << "Acc_test " << "gamma "<< "lambda" << endl;
    for (double g : gamma_list){
        for (double lam: Lambda_list){
            LogReg my_model(g, classes, X_train, y_train, etaa, lam, epochss, batch_szz, num_train, features);
            my_model.fit();
            accuracy_v = my_model.compute_accuracy(X_val, y_val, num_val);
            accuracy_t = my_model.compute_accuracy(X_test, y_test, num_test);

            ofile << accuracy_v << " " << accuracy_t << " " << g << " " << lam << endl;


        }
    }
    ofile.close();
}

void ADAM_hyperparams(int classes, mat X_train, mat y_train, int num_train, int features, mat X_test, mat y_test, int num_test, mat X_val, mat y_val, int num_val){

    double Lambda = 1e-4;
    double accuracy_t, accuracy_v;
    double epsilon = 1e-8;


    int epochss = 100;
    int batch_szz = 500;
    double etaa = 0.1;
    double beta1_list[6] = {0.80, 0.85, 0.90, 0.95, 0.99, 0.999};
    double beta2_list[6] = {0.80, 0.85, 0.90, 0.95, 0.99, 0.999};

    ofstream ofile;
    ofile.open("ADAM_LogReg_final.txt");
    ofile << "Acc_val " << "Acc_test " << "beta1 " << "beta2" << endl;
    for (double beta1 : beta1_list){
        for (double beta2 : beta2_list){

            LogReg my_model(beta1, beta2, epsilon, classes, X_train, y_train, etaa, Lambda, epochss, batch_szz, num_train, features);
            my_model.fit();
            accuracy_v = my_model.compute_accuracy(X_val, y_val, num_val);
            accuracy_t = my_model.compute_accuracy(X_test, y_test, num_test);

            ofile << accuracy_v << " " << accuracy_t << " " << beta1 << " " << beta2 << endl;

        }
    }

    ofile.close();
}

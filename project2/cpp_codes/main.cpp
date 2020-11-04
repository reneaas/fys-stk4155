#include "neural_network.hpp"
#include <time.h>
#include <cstdio>
#include <fstream>
#include <cmath>

void read_mnist(char* filename_X, char* filename_Y, double *X_train, double *y_train, int N_train, int features, int n_classes);
void test_mnist(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, double lambda, double gamma, int features, int num_points, string outfilename);
void write_to_file(double x, double y, double z, string outfilename);
void read_franke(char* infilename, double *X_train, double *X_val, double *X_test, double *y_train, double *y_val, double *y_test, int N, int features, int n_train, int n_val, int n_test);
void test_franke(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, double lambda, double gamma, int features, int num_points, string outfilename);

int main(int argc, char const *argv[]) {
    int hidden_layers = atoi(argv[1]);
    int nodes = atoi(argv[2]);
    int num_outputs = atoi(argv[3]);
    int epochs = atoi(argv[4]);
    int batch_size = atoi(argv[5]);
    double eta = atof(argv[6]);
    double lambda = atof(argv[7]);
    double gamma = atof(argv[8]);
    int features = atoi(argv[9]);
    int num_points = atoi(argv[10]);
    string outfilename = argv[11];


    //test_mnist(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, lambda, gamma, features, num_points, outfilename);
    test_franke(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, lambda, gamma, features, num_points, outfilename);

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


void test_mnist(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, double lambda, double gamma, int features, int num_points, string outfilename)
{
    string problem_type = "classification";
    string hidden_act = "sigmoid";
    double *X = new double[num_points*features];
    double *y = new double[num_points*num_outputs];
    char* filename_X = "./data_files/mnist_training_X.txt";
    char* filename_Y = "./data_files/mnist_training_Y.txt";
    read_mnist(filename_X, filename_Y, X, y, num_points, features, num_outputs);

    //FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type); //Default to sigmoid for hidden activations
    //FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type, hidden_act); //No regularization or momentum
    //FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type, hidden_act, lambda); //L2 regularization, no momentum
    FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type, hidden_act, lambda, gamma); //L2 reg + momentum

    //cout << num_points << endl;
    int n_train = 0.95*num_points;
    int n_validate = 0.05*num_points;
    cout << n_train << endl;
    cout << n_validate << endl;

    double *X_train = new double[n_train*features];
    double *y_train = new double[n_train*num_outputs];

    double *X_validate = new double[n_validate*features];
    double *y_validate = new double[n_validate*num_outputs];

    //extract training data
    for (int i = 0; i < n_train; i++){
        for (int j = 0; j < features; j++){
            X_train[i*features + j] = X[i*features + j];
        }
    }

    for (int i = 0; i < n_train; i++){
        for (int j = 0; j < num_outputs; j++){
            y_train[i*num_outputs + j] = y[i*num_outputs + j];
        }
    }

    //Extract validation data
    for (int i = 0; i < n_validate; i++){
        for (int j = 0; j < features; j++){
            X_validate[i*features + j] = X[(i+n_train)*features + j];
        }
    }

    for (int i = 0; i < n_validate; i++){
        for (int j = 0; j < num_outputs; j++){
            y_validate[i*num_outputs + j] = y[(i+n_train)*num_outputs + j];
        }
    }

    my_network.init_data(X_train, y_train, n_train);
    clock_t start = clock();
    my_network.fit();
    clock_t end = clock();
    double timeused = (double)(end-start)/CLOCKS_PER_SEC;
    cout << "CPU time = " << timeused << " seconds " << endl;

    //Validate model:
    double accuracy = my_network.evaluate(X_validate, y_validate, n_validate);
    write_to_file((double) eta, (double) num_points, accuracy, outfilename);



    /*
    int N_test = 10000;
    double *X_test = new double[N_test*features];
    double *y_test = new double[N_test*num_outputs];
    filename_X = "./data_files/mnist_test_X.txt";
    filename_Y = "./data_files/mnist_test_Y.txt";
    read_mnist(filename_X, filename_Y, X_test, y_test, N_test, features, num_outputs);
    double accuracy = my_network.evaluate(X_test, y_test, N_test);
    write_to_file((double) eta, (double) num_points, accuracy, outfilename);
    */
}


void write_to_file(double x, double y, double z, string outfilename)
{
    ofstream ofile;
    ofile.open(outfilename);
    ofile << x << " " << y << " " << z << endl;
    ofile.close();
}

void test_franke(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, double lambda, double gamma, int features, int num_points, string outfilename)
{
    string problem_type = "regression";
    string hidden_act = "sigmoid";
    double *X_train, *X_val, *X_test, *y_train, *y_val, *y_test;
    char *infilename = "./data_files/frankefunction_dataset_N_10000.txt";
    double fraction_train = 0.9*0.95;
    double fraction_val = 0.9*0.05;
    double fraction_test = 0.1;

    int n_train = fraction_train*num_points;
    int n_val = fraction_val*num_points;
    int n_test = fraction_test*num_points;

    X_train = new double[features*n_train];
    X_val = new double[features*n_val];
    X_test = new double[features*n_test];

    y_train = new double[n_train];
    y_val = new double[n_val];
    y_test = new double[n_test];

    read_franke(infilename, X_train, X_val, X_test, y_train, y_val, y_test, num_points, features, n_train, n_val, n_test);
    FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type); //Default to sigmoid for hidden activations
    //FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type, hidden_act); //No regularization or momentum
    //FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type, hidden_act, lambda); //L2 regularization, no momentum
    //FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type, hidden_act, lambda, gamma); //L2 reg + momentum

    my_network.init_data(X_train, y_train, n_train);
    clock_t start = clock();
    my_network.fit();
    clock_t end = clock();
    double timeused = (double)(end-start)/CLOCKS_PER_SEC;
    cout << "CPU time = " << timeused << " seconds " << endl;

    //Validate model:
    double r2 = my_network.evaluate(X_val, y_val, n_val);
    //write_to_file((double) eta, (double) num_points, r2, outfilename);




}

void read_franke(char* infilename, double *X_train, double *X_val, double *X_test, double *y_train, double *y_val, double *y_test, int N, int features, int n_train, int n_val, int n_test)
{
    double *x = new double[N];
    double *y = new double[N];
    double *z = new double[N];

    double x_mean = 0., x_std = 0.;
    double y_mean = 0., y_std = 0.;
    double z_mean = 0., z_std = 0.;

    for (int i = 0; i < N; i++){
        x_mean += x[i];
        y_mean += y[i];
        z_mean += z[i];
    }

    x_mean *= (1./N);
    y_mean *= (1./N);
    z_mean *= (1./N);

    for (int i = 0; i < N; i++){
        x_std += (x[i]-x_mean)*(x[i]-x_mean);
        y_std += (y[i]-y_mean)*(y[i]-y_mean);
        z_std += (z[i]-z_mean)*(z[i]-z_mean);
    }
    x_std *= (1./(N-1));
    y_std *= (1./(N-1));
    z_std *= (1./(N-1));

    x_std = sqrt(x_std);
    y_std = sqrt(y_std);
    z_std = sqrt(z_std);

    for (int i = 0; i < N; i++){
        x[i] = (x[i]-x_mean)/x_std;
        y[i] = (y[i]-y_mean)/y_std;
        z[i] = (z[i]-z_mean)/z_std;
    }


    FILE *fp_data = fopen(infilename, "r");


    for (int i = 0; i < N; i++){
        fscanf(fp_data, "%lf %lf %lf", &x[i], &y[i], &z[i]);
    }
    fclose(fp_data);

    for (int i = 0; i < n_train; i++){
        X_train[i*features] = 1.;
        y_train[i] = z[i];
    }

    for (int i = 0; i < n_val; i++){
        X_val[i*features] = 1.;
        y_val[i] = y[n_train + i];
    }

    for (int i = 0; i < n_test; i++){
        X_test[i*features] = 1.;
        y_test[i] = y[n_train + n_val + i];
    }

    //Set up design matrices
    int max_deg = 0;
    int col_idx = 1;

    while (col_idx < features){
        max_deg++;
        for (int i = 0; i < n_train; i++){
            for (int j = 0; j < max_deg + 1; j++){
                X_train[i*features + col_idx] = pow(x[i], max_deg - j)*pow(y[i], j);
            }
        }

        for (int i = 0; i < n_val; i++){
            for (int j = 0; j < max_deg + 1; j++){
                X_val[i*features + col_idx] = pow(x[n_train + i],  max_deg-j)*pow(y[n_train + i], j);
            }
        }

        for (int i = 0; i < n_test; i++){
            for (int j = 0; j < max_deg + 1; j++){
                X_test[i*features + col_idx] = pow(x[n_train + n_val + i], max_deg-j)*pow(y[n_train + n_val + i], j);
            }
        }
        col_idx++;
    }

    delete[] x;
    delete[] y;
    delete[] z;
}

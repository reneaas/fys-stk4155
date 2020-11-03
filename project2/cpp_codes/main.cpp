#include "neural_network.hpp"
#include <time.h>
#include <cstdio>
#include <fstream>

void read_mnist(char* filename_X, char* filename_Y, double *X_train, double *y_train, int N_train, int features, int n_classes);
void test_mnist(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, double lambda, double gamma, int features, int num_points, string outfilename);
void write_to_file(double x, double y, double z, string outfilename);

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
    test_mnist(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, lambda, gamma, features, num_points, outfilename);

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
    double *X_train = new double[num_points*features];
    double *y_train = new double[num_points*num_outputs];
    char* filename_X = "./data_files/mnist_training_X.txt";
    char* filename_Y = "./data_files/mnist_training_Y.txt";
    read_mnist(filename_X, filename_Y, X_train, y_train, num_points, features, num_outputs);

    FFNN my_network(hidden_layers, nodes, num_outputs, epochs, batch_size, eta, features, problem_type); //Default to sigmoid for hidden activations
    //FFNN my_network(hidden_layers, nodes, N_outputs, epochs, batch_size, eta, features, problem_type, hidden_act); //No regularization or momentum
    //FFNN my_network(hidden_layers, nodes, N_outputs, epochs, batch_size, eta, features, problem_type, hidden_act, lambda); //L2 regularization, no momentum
    //FFNN my_network(hidden_layers, nodes, N_outputs, epochs, batch_size, eta, features, problem_type, hidden_act, lambda, gamma); //L2 reg + momentum

    my_network.init_data(X_train, y_train, num_points);
    clock_t start = clock();
    my_network.fit();
    clock_t end = clock();
    double timeused = (double)(end-start)/CLOCKS_PER_SEC;
    cout << "CPU time = " << timeused << " seconds " << endl;



    int N_test = 10000;
    double *X_test = new double[N_test*features];
    double *y_test = new double[N_test*num_outputs];
    filename_X = "./data_files/mnist_test_X.txt";
    filename_Y = "./data_files/mnist_test_Y.txt";
    read_mnist(filename_X, filename_Y, X_test, y_test, N_test, features, num_outputs);
    double accuracy = my_network.evaluate(X_test, y_test, N_test);

    write_to_file((double) eta, (double) num_points, accuracy, outfilename);
}


void write_to_file(double x, double y, double z, string outfilename)
{
    ofstream ofile;
    ofile.open(outfilename);
    ofile << x << " " << y << " " << z << endl;
    ofile.close();
}

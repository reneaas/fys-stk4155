
void create_mnist_binary_files(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int num_train, int num_val, int num_test);


int main(int argc, char const *argv[]) {

    return 0;
}

void create_mnist_binary_files(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test, int num_train, int num_val, int num_test){
    int features = 28*28;
    int num_outputs = 10;


    char* infilename_X = "./data_files/mnist_training_X.txt";
    char* infilename_y = "./data_files/mnist_training_Y.txt";
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

    (*X_train).save("../data_files/mnist_X_train.bin");
    (*y_train).save("../data_files/mnist_y_train.bin");

    (*X_val).save("../data_files/mnist_X_val.bin");
    (*y_val).save("../data_files/mnist_y_val.bin");




    infilename_X = "../data_files/mnist_test_X.txt";
    infilename_y = "../data_files/mnist_test_Y.txt";
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

    (*X_test).save("../data_files/mnist_X_test.bin");
    (*y_test).save("../data_files/mnist_y_test.bin");

}

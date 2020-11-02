#include "neural_network.hpp"

FFNN::FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features)
{
    //Specify parameters of the model
    layers_ = hidden_layers+2;
    nodes_ = nodes;
    epochs_ = epochs;
    batch_size_ = batch_size;
    eta_ = eta*(1./batch_size_);
    features_ = features;
    num_outputs_ = num_outputs;

    num_rows_ = new int[layers_-1]();
    num_cols_ = new int[layers_-1]();
    r_w_ = new int[layers_]();
    r_a_ = new int[layers_ + 1]();
    r_b_ = new int[layers_]();

    y_ = new double[num_outputs_]();


    num_rows_[0] = nodes;
    num_cols_[0] = features;
    num_rows_[layers_ - 2] = num_outputs_;
    num_cols_[layers_ - 2] = nodes;

    for (int l = 1; l < layers_-2; l++){
        num_rows_[l] = nodes;
        num_cols_[l] = nodes;
    }

    for (int l = 0; l < layers_-1; l++){
        r_w_[l+1] = r_w_[l] + num_rows_[l]*num_cols_[l];
        r_b_[l+1] = r_b_[l] + num_rows_[l];
        r_a_[l+1] = r_a_[l] + num_cols_[l];
    }
    r_a_[layers_] = r_a_[layers_-1] + num_outputs_;


    weights_ = new double[r_w_[layers_-1]]();
    biases_ = new double[r_b_[layers_-1]]();
    activations_ = new double[r_a_[layers_]]();
    error_ = new double[r_b_[layers_-1]]();

    //Initialize weights and biases in hidden layers.
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1); //Normal distribution(mu, std)
    //Initialize weights and biases
    int rows, cols, stride_w, stride_b;
    double sqrt_cols_inv;
    for (int l = 0; l < layers_-1; l++){
        rows = num_rows_[l];
        cols = num_cols_[l];
        stride_w = r_w_[l];
        stride_b = r_b_[l];
        sqrt_cols_inv = 1./sqrt(cols);

        for (int j = 0; j < rows; j++){
            biases_[stride_b + j] = distribution(generator);
            for (int k = 0; k < cols; k++){
                weights_[stride_w + j*cols + k] = distribution(generator)*sqrt_cols_inv;
            }
        }
    }
}

//miscellaneous functions
double FFNN::sigmoid(double z)
{
    return 1./(1. + exp(-z));
}

void FFNN::init_data(double *X_data, double *y_data, int num_points)
{
    num_points_ = num_points;
    X_data_ = X_data;
    y_data_ = y_data;
}


void FFNN::fit()
{
    int num_batches = num_points_/batch_size_;
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, num_points_-1);
    int idx;
    dw_ = new double[r_w_[layers_-1]]();
    db_ = new double[r_b_[layers_-1]]();

    for (int epoch = 0; epoch < epochs_; epoch++){

        cout << "epoch = " << epoch << " of " << epochs_ <<  "\r" << endl;

        for (int batch = 0; batch < num_batches; batch++){
            //Zero out gradients
            for (int b = 0; b < batch_size_; b++){
                //Extract datapoint
                idx = distribution(generator);
                for (int j = 0; j < features_; j++){
                    activations_[j] = X_data_[idx*features_ + j];
                }
                for (int j = 0; j < num_outputs_; j++){
                    y_[j] = y_data_[idx*num_outputs_ + j];
                }

                feed_forward();
                backward_pass();
            }
            update_parameters();
        }
    }
}

void FFNN::feed_forward()
{
    double z;
    int rows, cols, stride_w, stride_b, stride_ap, stride_a;
    int l, j, k;
    //Compute activations through the hidden layers
    for (l = 1; l < layers_-1; l++){
        rows = num_rows_[l-1];
        cols = num_cols_[l-1];
        stride_w = r_w_[l-1];
        stride_b = r_b_[l-1];
        stride_ap = r_a_[l-1];
        stride_a = r_a_[l];

        for (j = 0; j < rows; j++){
            z = biases_[stride_b + j];
            for (k = 0; k < cols; k++){
                z += weights_[stride_w + j*cols + k]*activations_[stride_ap + k];
            }
            activations_[stride_a + j] = sigmoid(z);
        }
    }


    //Compute activations at top layer:
    l = layers_-1;
    rows = num_rows_[l-1];
    cols = num_cols_[l-1];
    stride_w = r_w_[l-1];
    stride_b = r_b_[l-1];
    stride_ap = r_a_[l-1];
    stride_a = r_a_[l];

    double Z = 0.;
    for (j = 0; j < rows; j++){
        z = biases_[stride_b + j];
        for (k = 0; k < cols; k++){
            z += weights_[stride_w + j*cols + k]*activations_[stride_ap + k];
        }
        z = exp(z);
        activations_[stride_a + j] = z;
        Z += z;
    }
    //Compute softmax:
    for (j = 0; j < rows; j++){
        activations_[stride_a + j] /= Z;
    }
}


void FFNN::backward_pass(){
    double delta = 0.;
    double tmp;
    int rows, cols, stride_w, stride_b, stride_a, stride_ap;
    int l, j, k;

    //Compute error in top layer
    l = layers_-2;
    rows = num_rows_[l];
    cols = num_cols_[l];
    stride_w = r_w_[l];
    stride_b = r_b_[l];
    stride_a = r_a_[l+1];
    stride_ap = r_a_[l];
    for (j = 0; j < num_outputs_; j++){
        error_[stride_b + j] = activations_[stride_a + j] - y_[j];
    }

    //Update gradients in top layer
    for (j = 0; j < rows; j++){
        tmp = error_[stride_b + j];
        db_[stride_b + j] += tmp;
        for (k = 0; k < cols; k++){
            dw_[stride_w + j*cols + k] += tmp*activations_[stride_ap + k];
        }
    }


    //Compute error and update gradients in remaining layers
    double s;
    int stride_bp;
    for (l = layers_ - 2; l > 0; l--){
        rows = num_rows_[l];
        cols = num_cols_[l];
        stride_w = r_w_[l];
        stride_b = r_b_[l];
        stride_bp = r_b_[l-1];
        stride_a = r_a_[l];

        //Backpropagate the error from layers l+1 to layer l.
        for (j = 0; j < rows; j++){
            s = activations_[stride_a+j];
            s = s*(1-s);
            tmp = 0.;
            for (k = 0; k < cols; k++){
                tmp +=  weights_[stride_w + k*cols + j]*error_[stride_b + k];
            }
            error_[stride_bp + j] = tmp*s;
        }


        //Update gradients
        rows = num_rows_[l-1];
        cols = num_cols_[l-1];
        stride_w = r_w_[l-1];
        stride_b = r_b_[l-1];
        stride_a = r_a_[l-1];
        for (j = 0; j < rows; j++){
            tmp = error_[stride_b + j];
            db_[stride_b + j] += tmp;
            for (k = 0; k < cols; k++){
                dw_[stride_w + j*cols + k] += tmp*activations_[stride_a + k];
            }
        }
    }
}


void FFNN::update_parameters()
{
    int l, j, k;
    int rows, cols, stride_w, stride_b;

    for (l = 0; l < layers_-1; l++){
        rows = num_rows_[l];
        cols = num_cols_[l];
        stride_w = r_w_[l];
        stride_b = r_b_[l];

        for (j = 0; j < rows; j++){
            biases_[stride_b + j] -= eta_*db_[stride_b + j];
            db_[stride_b + j] = 0.;
            for (k = 0; k < cols; k++){
                weights_[stride_w + j*cols + k] -= eta_*dw_[stride_w + j*cols + k];
                dw_[stride_w + j*cols + k] = 0.;
            }
        }
    }

}

void FFNN::predict(double *X_test, double *y_test, int num_tests)
{
    double correct_predictions = 0.;
    double wrong_predictions = 0.;
    int stride_a = r_a_[layers_ - 1];

    for (int i = 0; i < num_tests; i++){
        for (int j = 0; j < features_; j++){
            activations_[j] = X_test[i*features_ + j];
        }

        for (int j = 0; j < num_outputs_; j++){
            y_[j] = y_test[i*num_outputs_ + j];
        }

        feed_forward();

        int max_idx;
        double max_elem = 0.;

        for (int j = 0; j < num_outputs_; j++){
            if (activations_[stride_a + j] > max_elem){
                max_elem = activations_[stride_a + j];
                max_idx = j;
            }
        }

        if (y_[max_idx] == 1){
            correct_predictions += 1;
        }
        else{
            wrong_predictions += 1;
        }

    }

    double accuracy = correct_predictions*(1./num_tests);
    cout << "Correct predictions = " << correct_predictions << " of " << num_tests << endl;
    cout << "Wrong predictions = " << wrong_predictions << " of " << num_tests << endl;
    cout << "Accuracy = " << accuracy << endl;

    delete[] X_test;
    delete[] y_test;
}


FFNN::~FFNN()
{
delete[] X_data_;
delete[] y_data_;
delete[] weights_;
delete[] biases_;
delete[] activations_;
delete[] y_;
delete[] num_rows_;
delete[] num_cols_;
delete[] r_w_;
delete[] r_b_;
delete[] r_a_;
delete[] dw_;
delete[] db_;
delete[] error_;
}

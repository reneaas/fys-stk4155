#include "neural_network.hpp"

FFNN::FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type)
{
    if (problem_type == "classification"){
        top_layer_act = &FFNN::predict_softmax;
    }
    if (problem_type == "regression"){
        top_layer_act = &FFNN::predict_linear;
    }

    hidden_act = &FFNN::sigmoid; //Default
    hidden_act_derivative = &FFNN::sigmoid_derivative;
    update_parameters = &FFNN::update;

    //Specify parameters of the model
    layers_ = hidden_layers+2;
    nodes_ = nodes;
    epochs_ = epochs;
    batch_size_ = batch_size;
    eta_ = eta;
    features_ = features;
    num_outputs_ = num_outputs;

    create_model_arch();
    init_parameters();


    gamma_ = 0; //Must be set to zero to avoid an attempt to free a few pointers layer on.
}

FFNN::FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type, string hidden_activation){
    if (problem_type == "classification"){
        top_layer_act = &FFNN::predict_softmax;
    }
    if (problem_type == "regression"){
        top_layer_act = &FFNN::predict_linear;
    }

    if (hidden_activation == "sigmoid"){
        hidden_act = &FFNN::sigmoid;
        hidden_act_derivative = &FFNN::sigmoid_derivative;
    }

    if (hidden_activation == "relu"){
        hidden_act = &FFNN::relu;
        hidden_act_derivative = &FFNN::relu_derivative;
    }

    if (hidden_activation == "leaky_relu"){
        hidden_act = &FFNN::leaky_relu;
        hidden_act_derivative = &FFNN::leaky_relu_derivative;
    }

    update_parameters = &FFNN::update;

    //Specify parameters of the model
    layers_ = hidden_layers+2;
    nodes_ = nodes;
    epochs_ = epochs;
    batch_size_ = batch_size;
    eta_ = eta;
    features_ = features;
    num_outputs_ = num_outputs;

    create_model_arch();
    init_parameters();


    gamma_ = 0; //Must be set to zero to avoid an attempt to free a few pointers layer on.
}

FFNN::FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type, string hidden_activation, double lambda)
{
    if (problem_type == "classification"){
        top_layer_act = &FFNN::predict_softmax;
    }
    if (problem_type == "regression"){
        top_layer_act = &FFNN::predict_linear;
    }

    if (hidden_activation == "sigmoid"){
        hidden_act = &FFNN::sigmoid;
        hidden_act_derivative = &FFNN::sigmoid_derivative;
    }

    if (hidden_activation == "relu"){
        hidden_act = &FFNN::relu;
        hidden_act_derivative = &FFNN::relu_derivative;
    }

    if (hidden_activation == "leaky_relu"){
        hidden_act = &FFNN::leaky_relu;
        hidden_act_derivative = &FFNN::leaky_relu_derivative;
    }


    //Specify parameters of the model
    layers_ = hidden_layers+2;
    nodes_ = nodes;
    epochs_ = epochs;
    batch_size_ = batch_size;
    eta_ = eta;
    features_ = features;
    num_outputs_ = num_outputs;
    lambda_ = lambda;

    create_model_arch();
    init_parameters();

    if (lambda_ > 0){
        update_parameters = &FFNN::update_l2;
    }

    gamma_ = 0; //Must be set to zero to avoid an attempt to free a few pointers layer on.
}

FFNN::FFNN(int hidden_layers, int nodes, int num_outputs, int epochs, int batch_size, double eta, int features, string problem_type, string hidden_activation, double lambda, double gamma)
{
    if (problem_type == "classification"){
        top_layer_act = &FFNN::predict_softmax;
    }
    if (problem_type == "regression"){
        top_layer_act = &FFNN::predict_linear;
    }

    if (hidden_activation == "sigmoid"){
        hidden_act = &FFNN::sigmoid;
        hidden_act_derivative = &FFNN::sigmoid_derivative;
    }

    if (hidden_activation == "relu"){
        hidden_act = &FFNN::relu;
        hidden_act_derivative = &FFNN::relu_derivative;
    }

    if (hidden_activation == "leaky_relu"){
        hidden_act = &FFNN::leaky_relu;
        hidden_act_derivative = &FFNN::leaky_relu_derivative;
    }


    //Specify parameters of the model
    layers_ = hidden_layers+2;
    nodes_ = nodes;
    epochs_ = epochs;
    batch_size_ = batch_size;
    eta_ = eta;
    features_ = features;
    num_outputs_ = num_outputs;
    lambda_ = lambda;
    gamma_ = gamma;

    create_model_arch();
    init_parameters();

    update_parameters = &FFNN::update_momentum_l2;
    vb_ = new double[r_b_[layers_-1]]();
    vw_ = new double[r_w_[layers_-1]]();
}



void FFNN::create_model_arch()
{
    num_rows_ = new int[layers_-1]();
    num_cols_ = new int[layers_-1]();
    r_w_ = new int[layers_]();
    r_a_ = new int[layers_ + 1]();
    r_b_ = new int[layers_]();

    y_ = new double[num_outputs_]();
    num_rows_[0] = nodes_;
    num_cols_[0] = features_;
    num_rows_[layers_ - 2] = num_outputs_;
    num_cols_[layers_ - 2] = nodes_;

    for (int l = 1; l < layers_-2; l++){
        num_rows_[l] = nodes_;
        num_cols_[l] = nodes_;
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
    z_ = new double[r_a_[layers_]]();
    error_ = new double[r_b_[layers_-1]]();
}


void FFNN::init_parameters()
{
    default_random_engine generator;
    normal_distribution<double> distribution(0., 1.); //Normal distribution(mu, std)
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
        cout << "epoch = " << epoch+1 << " of " << epochs_ << endl;

        for (int batch = 0; batch < num_batches; batch++){
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
            (this->*update_parameters)();
        }
    }


    //Clear up memory after the model is fitted.
    if (gamma_ > 0){
        delete[] vb_;
        delete[] vw_;
    }
    delete[] dw_;
    delete[] db_;
    delete[] X_data_;
    delete[] y_data_;
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
            activations_[stride_a + j] = (this->*hidden_act)(z);
            z_[stride_a + j] = z;
        }
    }
    (this->*top_layer_act)();
}


void FFNN::backward_pass(){
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
    for (l = layers_-2; l > 0; l--){
        rows = num_rows_[l-1]; //Number of rows of layer l-1
        cols = num_cols_[l]; //Number of columns of layer l
        stride_w = r_w_[l]; //Number of elems to skip in the weights array.
        stride_b = r_b_[l]; //Number of elems to skip in the bias array
        stride_bp = r_b_[l-1]; //Number of elems to skip in the bias array to update error at layer l-1
        stride_a = r_a_[l]; //Number of elems to skip in the activations array.

        //Backpropagate the error from layers l+1 to layer l.
        for (j = 0; j < rows; j++){
            //s = activations_[stride_a+j];
            //s = s*(1-s);

            s = z_[stride_a + j];
            s = (this->*hidden_act_derivative)(s);
            tmp = 0.;
            for (k = 0; k < cols; k++){
                tmp +=  weights_[stride_w + k*cols + j]*error_[stride_b + k];
            }
            //cout << tmp*s << endl;
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


void FFNN::update()
{
    int l, j, k;
    int rows, cols, stride_w, stride_b;
    double step = eta_/batch_size_;

    for (l = 0; l < layers_-1; l++){
        rows = num_rows_[l];
        cols = num_cols_[l];
        stride_w = r_w_[l];
        stride_b = r_b_[l];

        for (j = 0; j < rows; j++){
            biases_[stride_b + j] -= step*db_[stride_b + j];
            db_[stride_b + j] = 0.;
            for (k = 0; k < cols; k++){
                weights_[stride_w + j*cols + k] -= step*dw_[stride_w + j*cols + k];
                dw_[stride_w + j*cols + k] = 0.;
            }
        }
    }
}

void FFNN::update_l2()
{
    int l, j, k;
    int rows, cols, stride_w, stride_b;
    double step = eta_/batch_size_;

    for (l = 0; l < layers_-1; l++){
        rows = num_rows_[l];
        cols = num_cols_[l];
        stride_w = r_w_[l];
        stride_b = r_b_[l];

        for (j = 0; j < rows; j++){
            biases_[stride_b + j] -= step*db_[stride_b + j];
            db_[stride_b + j] = 0.;
            for (k = 0; k < cols; k++){
                weights_[stride_w + j*cols + k] -= step*(lambda_*weights_[stride_w + j*cols + k] + dw_[stride_w + j*cols + k]);
                dw_[stride_w + j*cols + k] = 0.;
            }
        }
    }
}

void FFNN::update_momentum_l2()
{
    int l, j, k;
    int rows, cols, stride_w, stride_b;
    double step = eta_/batch_size_;

    for (l = 0; l < layers_-1; l++){
        rows = num_rows_[l];
        cols = num_cols_[l];
        stride_w = r_w_[l];
        stride_b = r_b_[l];

        for (j = 0; j < rows; j++){
            vb_[stride_b + j] = gamma_*vb_[stride_b + j] + step*db_[stride_b + j];
            biases_[stride_b + j] -= vb_[stride_b + j];
            db_[stride_b + j] = 0.;
            for (k = 0; k < cols; k++){
                vw_[stride_w + j*cols + k] = gamma_*vw_[stride_w + j*cols + k] + step*(lambda_*weights_[stride_w + j*cols + k] + dw_[stride_w + j*cols + k]);
                weights_[stride_w + j*cols + k] -= vw_[stride_w + j*cols + k];
                dw_[stride_w + j*cols + k] = 0.;
            }
        }
    }
}

double FFNN::evaluate(double *X_test, double *y_test, int num_tests)
{
    double correct_predictions = 0.;
    double wrong_predictions = 0.;
    int stride_a = r_a_[layers_-1];

    for (int i = 0; i < num_tests; i++){
        for (int j = 0; j < features_; j++){
            activations_[j] = X_test[i*features_ + j];
        }

        for (int j = 0; j < num_outputs_; j++){
            y_[j] = y_test[i*num_outputs_ + j];
        }

        feed_forward();


        double max_elem = 0.;
        int max_idx;

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

    return accuracy;
}

/*
Top layer activation functions
*/

void FFNN::predict_linear()
{
    //Compute activations at top layer:
    int l = layers_-1;
    int rows = num_rows_[l-1];
    int cols = num_cols_[l-1];
    int stride_w = r_w_[l-1];
    int stride_b = r_b_[l-1];
    int stride_ap = r_a_[l-1];
    int stride_a = r_a_[l];
    double z;

    for (int j = 0; j < rows; j++){
        z = biases_[stride_b + j];
        for (int k = 0; k < cols; k++){
            z += weights_[stride_w + j*cols + k]*activations_[stride_ap + k];
        }
        activations_[stride_a + j] = z;
    }
}

void FFNN::predict_softmax()
{
    //Compute activations at top layer:
    int l = layers_-1;
    int rows = num_rows_[l-1];
    int cols = num_cols_[l-1];
    int stride_w = r_w_[l-1];
    int stride_b = r_b_[l-1];
    int stride_ap = r_a_[l-1];
    int stride_a = r_a_[l];
    double z;

    double Z = 0.;
    for (int j = 0; j < rows; j++){
        z = biases_[stride_b + j];
        for (int k = 0; k < cols; k++){
            z += weights_[stride_w + j*cols + k]*activations_[stride_ap + k];
        }
        z = exp(z);
        activations_[stride_a + j] = z;
        Z += z;
    }
    //Compute softmax:
    for (int j = 0; j < rows; j++){
        activations_[stride_a + j] /= Z;
    }
}

/*
Various activation functions
*/
double FFNN::sigmoid(double z)
{
    return 1./(1. + exp(-z));
}

double FFNN::sigmoid_derivative(double z)
{
    double s = 1./(1.+exp(-z));
    return s*(1-s);
}

double FFNN::relu(double z)
{
    return z*(z > 0);
}

double FFNN::relu_derivative(double z)
{
    return (z > 0);
}

double FFNN::leaky_relu(double z)
{
    return 0.1*z*(z <= 0) + z*(z > 0);
}

double FFNN::leaky_relu_derivative(double z)
{
    return 0.1*(z <= 0) + (z > 0);
}


FFNN::~FFNN()
{
    delete[] weights_;
    delete[] biases_;
    delete[] activations_;
    delete[] y_;
    delete[] num_rows_;
    delete[] num_cols_;
    delete[] r_w_;
    delete[] r_b_;
    delete[] r_a_;
    delete[] error_;
}

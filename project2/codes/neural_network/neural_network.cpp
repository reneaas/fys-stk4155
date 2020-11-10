#include "neural_network.hpp"


FFNN::FFNN(int hidden_layers, int features, int nodes, int num_outputs, string model_type, double lamb, double gamma, string hidden_activation)
{
    features_ = features;
    nodes_ = nodes;
    num_outputs_ = num_outputs;
    hidden_layers_ = hidden_layers;
    num_layers_ = hidden_layers_ + 1;

    //Set top layer activation function and which metric to use.
    if (model_type == "classification"){
        if (num_outputs == 1){
            top_layer_act = &FFNN::binary_classifier;
        }
        else{
            top_layer_act = &FFNN::softmax; //top layer activation
        }
        compute_metric = &FFNN::compute_accuracy;
    }
    else if (model_type == "regression"){
        top_layer_act = &FFNN::linear;
        compute_metric = &FFNN::compute_r2;
    }

    //Set hidden activation function
    if (hidden_activation == "sigmoid"){
        hidden_act = &FFNN::sigmoid;
        hidden_act_derivative = &FFNN::sigmoid_derivative;
    }
    else if (hidden_activation == "relu"){
        hidden_act = &FFNN::relu;
        hidden_act_derivative = &FFNN::relu_derivative;
    }
    else if (hidden_activation == "leaky_relu"){
        hidden_act = &FFNN::leaky_relu;
        hidden_act_derivative = &FFNN::leaky_relu_derivative;
    }

    gamma_ = gamma;
    lamb_ = lamb;

    if (gamma_ > 0){
        update_parameters = &FFNN::update_l2_momentum;

        //Add first hidden layer
        layers_.push_back(Layer(nodes_, features_, "sgd_momentum"));

        //Add hidden layers
        for (int l = 1; l < hidden_layers_; l++){
            layers_.push_back(Layer(nodes_, nodes_, "sgd_momentum"));
        }

        //Add top layer
        layers_.push_back(Layer(num_outputs_, nodes_, "sgd_momentum"));
    }
    else if (gamma_ == 0 && lamb_ > 0){
        update_parameters = &FFNN::update_l2;

        //Add first hidden layer
        layers_.push_back(Layer(nodes_, features_));

        //Add hidden layers
        for (int l = 1; l < hidden_layers_; l++){
            layers_.push_back(Layer(nodes_, nodes_));
        }

        //Add top layer
        layers_.push_back(Layer(num_outputs_, nodes_));
    }
    else{
        update_parameters = &FFNN::update;

        update_parameters = &FFNN::update_l2;

        //Add first hidden layer
        layers_.push_back(Layer(nodes_, features_));

        //Add hidden layers
        for (int l = 1; l < hidden_layers_; l++){
            layers_.push_back(Layer(nodes_, nodes_));
        }

        //Add top layer
        layers_.push_back(Layer(num_outputs_, nodes_));
    }
}



void FFNN::init_data(mat X_train, mat y_train, int num_points)
{
    num_points_ = num_points;
    X_train_ = X_train;
    y_train_ = y_train;
}

void FFNN::fit(int epochs, int batch_sz, double eta)
{
    batch_sz_ = batch_sz;
    eta_ = eta;
    //epochs_ = epochs;
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, num_points_-1);
    int idx;
    int batches = num_points_/batch_sz_;
    vec x, y;
    x = vec(features_);
    y = vec(num_outputs_);
    for (int epoch = 0; epoch < epochs; epoch++){
        cout << " epoch = " << epoch+1 << " of " << epochs << endl;

        for (int batch = 0; batch < batches; batch++){

            for (int b = 0; b < batch_sz_; b++){
                idx = distribution(generator);
                for (int j = 0; j < features_; j++){
                    x(j) = X_train_(j, idx);
                }
                for (int j = 0; j < num_outputs_; j++){
                    y(j) = y_train_(j, idx);
                }
                feed_forward(x);
                backward_pass(x, y);
            }
            (this->*update_parameters)();
        }
    }
}

/*
Basic update rule for the gradients with no regularization or momentum. Computed on a batch.
*/
void FFNN::update()
{
    double step = eta_*(1./batch_sz_);

    for (int l = 0; l < num_layers_; l++){
        layers_[l].weights_ -=  step*layers_[l].dw_;
        layers_[l].bias_ -= step*layers_[l].db_;
        layers_[l].dw_.fill(0.);
        layers_[l].db_.fill(0.);
    }
}

/*
Update rule for the gradients with regularization. No momentum. Computed on a batch.
*/
void FFNN::update_l2()
{
    double step = eta_*(1./batch_sz_);
    for (int l = 0; l < num_layers_; l++){
        layers_[l].weights_ -= step*(lamb_*layers_[l].weights_ + layers_[l].dw_);
        layers_[l].bias_ -= step*layers_[l].db_;
        layers_[l].dw_.fill(0.);
        layers_[l].db_.fill(0.);
    }
}

/*
Update rule for the gradients with regularization and momentum over a batch.
*/
void FFNN::update_l2_momentum()
{
    double step = eta_*(1./batch_sz_);
    for (int l = 0; l < num_layers_; l++){
        layers_[l].w_mom_ = gamma_*layers_[l].w_mom_ +  step*(lamb_*layers_[l].weights_ + layers_[l].dw_);
        layers_[l].weights_ -= layers_[l].w_mom_;
        layers_[l].b_mom_ = gamma_*layers_[l].b_mom_ + step*layers_[l].db_;
        layers_[l].bias_ -= layers_[l].b_mom_;
        layers_[l].dw_.fill(0.);
        layers_[l].db_.fill(0.);
    }
}

/*
Feed-forward part of the backpropagation algorithm
*/
void FFNN::feed_forward(vec x)
{
    //Process input activation
    layers_[0].z_ = layers_[0].weights_*x + layers_[0].bias_;
    layers_[0].activation_ = (this->*hidden_act)(layers_[0].z_);

    //Hidden layer activations
    for (int l = 1; l < hidden_layers_; l++){
        layers_[l].z_ = layers_[l].weights_*layers_[l-1].activation_ + layers_[l].bias_;
        layers_[l].activation_ = (this->*hidden_act)(layers_[l].z_);
    }

    //Top layer activation
    int l = num_layers_-1;
    layers_[l].z_ = layers_[l].weights_*layers_[l-1].activation_ + layers_[l].bias_;
    layers_[l].activation_ = (this->*top_layer_act)(layers_[l].z_);
}

/*
Backward pass of the backpropagation algorithm
*/
void FFNN::backward_pass(vec x, vec y)
{
    //Top layer error
    int l = num_layers_-1;
    layers_[l].error_ = layers_[l].activation_ - y;
    add_gradients(l);


    //Error in hidden layers
    for (int l = num_layers_ - 2; l > 0 ; l--){
        layers_[l].error_ = (layers_[l+1].weights_.t() * layers_[l+1].error_) % (this->*hidden_act_derivative)(layers_[l].z_);
        add_gradients(l);
    }

    //Compute gradients in first layer.
    l = 0;
    layers_[l].error_ = (layers_[l+1].weights_.t() * layers_[l+1].error_) % (this->*hidden_act_derivative)(layers_[l].z_);
    layers_[l].db_ += layers_[l].error_;
    layers_[l].dw_ += layers_[l].error_*x.t();

}

/*
Evaluate the model on validation or test data
*/
double FFNN::evaluate(mat X_test, mat y_test, int num_test){
    X_test_ = X_test;
    y_test_ = y_test;
    num_test_ = num_test;

    double metric = (this->*compute_metric)();
    return metric;
}

/*
Computes accuracy on the test set. Pointer to by compute_metric if model_type = "classification".
*/
double FFNN::compute_accuracy(){
    vec x = vec(features_);
    vec y = vec(num_outputs_);

    int idx;
    double accuracy = 0;
    double correct_predictions = 0;
    double wrong_predictions = 0;
    int l = num_layers_-1;
    for (int i = 0; i < num_test_; i++){
        for (int j = 0; j < features_; j++){
            x(j) = X_test_(j, i);
        }
        for (int j = 0; j < num_outputs_; j++){
            y(j) = y_test_(j, i);
        }
        feed_forward(x);
        idx = index_max(layers_[l].activation_);
        if (y(idx) == 1){
            correct_predictions++;
        }
        else{
            wrong_predictions++;
        }
    }
    accuracy = correct_predictions*(1./num_test_);
    return accuracy;
}

/*
Computes R2 score on a test set. Pointed to by compute_metric if model_type = "regression"
*/
double FFNN::compute_r2()
{
    vec x = vec(features_);
    vec y = vec(num_outputs_);
    double diff;

    double error = 0.;
    double y_mean = 0.;
    int l = num_layers_-1;
    for (int i = 0; i < num_test_; i++){
        for (int j = 0; j < features_; j++){
            x(j) = X_test_(j, i);
        }
        y(0) = y_test_(0, i);
        feed_forward(x);

        diff = y(0) - layers_[l].activation_(0);
        error += diff*diff;
        y_mean += y(0);
    }
    y_mean *= (1./num_test_);
    double tmp = 0;

    for (int j = 0; j < num_test_; j++){
        tmp += (y_test_(0, j)-y_mean)*(y_test_(0, j)-y_mean);
    }
    double r2 = 1 - error/tmp;
    return r2;
}

/*
Computes mse, but is by default no used.
Can manually be changed in the constructor.
*/
double FFNN::compute_mse()
{
    vec x = vec(features_);
    vec y = vec(num_outputs_);

    double mse = 0.;
    double diff;
    int l = num_layers_-1;
    for (int i = 0; i < num_test_; i++){
        for (int j = 0; j < features_; j++){
            x(j) = X_test_(j, i);
        }
        y(0) = y_test_(0, i);
        feed_forward(x);

        diff = y(0) - layers_[l].activation_(0);
        mse += diff*diff;
    }
    mse *= (1./num_test_);
    return mse;
}


void FFNN::add_gradients(int l)
{
    layers_[l].db_ += layers_[l].error_;
    layers_[l].dw_ += layers_[l].error_*layers_[l-1].activation_.t(); //outer product
}

/*
Hidden layer activation functions
*/
vec FFNN::sigmoid(vec z)
{
    return 1./(1. + exp(-z));
}

vec FFNN::sigmoid_derivative(vec z)
{
    vec res = 1./(1. + exp(-z));
    return res % (1-res);
}


vec FFNN::relu(vec z)
{
    vec s = z;
    return s.transform( [](double val){return val*(val > 0);});
}

vec FFNN::relu_derivative(vec z)
{
    vec s = z;
    return s.transform( [](double val){return (val > 0);});
}

vec FFNN::leaky_relu(vec z)
{
    vec s = z;
    return s.transform( [](double val){return 0.01*val*(val <= 0) + val*(val > 0);});
}

vec FFNN::leaky_relu_derivative(vec z)
{
    vec s = z;
    return s.transform( [](double val){return 0.01*(val <= 0) + (val > 0);});
}

/*
Top layer activation functions
*/
vec FFNN::softmax(vec a)
{
    vec res = exp(a);
    return exp(a)/sum(res);
}

vec FFNN::linear(vec z)
{
    return z;
}

vec FFNN::binary_classifier(vec z)
{
    vec s = 1./(1.+exp(z));
    return s.transform( [](double val){return (val > 0.5);});
}

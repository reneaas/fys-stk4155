#include "LogReg.hpp"


/* Constructor that initializes the model with the SGD optimizer */
LogReg::LogReg(int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features)
{

    M_ = classes;
    epochs_ = epochs;
    features_ = features;
    Lambda_ = Lambda;

    num_train_ = num_train;
    X_train_ = X_train;
    y_train_ = y_train;
    batch_sz_ = batch_sz;
    eta_ = eta/batch_sz_;
    z_ = vec(M_).fill(0.);

    weights_ = randn<mat>(M_, features_)*(1./sqrt(features_));
    dw_ = mat(M_, features_).fill(0.);

    bias_ = randn<vec>(M_);
    db_ = vec(M_).fill(0.);

    output_ = vec(M_).fill(0.);

    activation_ = &LogReg::softmax;
    optimizer_ = &LogReg::SGD;

}

/* Constructor that initializes the model with the SGD with momentum optimizer */
LogReg::LogReg(double gamma, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features){

    M_ = classes;
    epochs_ = epochs;
    features_ = features;
    Lambda_ = Lambda;
    gamma_ = gamma;

    num_train_ = num_train;
    X_train_ = X_train;
    y_train_ = y_train;
    batch_sz_ = batch_sz;
    eta_ = eta/batch_sz_;
    z_ = vec(M_).fill(0.);

    weights_ = randn<mat>(M_, features_)*(1./sqrt(features_));
    tmp_weights_ = mat(M_, features_).fill(0.);
    dw_ = mat(M_, features_).fill(0.);

    bias_ = randn<vec>(M_);
    tmp_bias_ = randn<vec>(M_);
    db_ = vec(M_).fill(0.);

    output_ = vec(M_).fill(0.);

    activation_ = &LogReg::softmax;
    optimizer_ = &LogReg::SGD_momentum;
}

/* Constructor that initializes the model with the Adam optimizer */
LogReg::LogReg(double beta1, double beta2, double epsilon, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features){

    M_ = classes;
    epochs_ = epochs;
    features_ = features;
    Lambda_ = Lambda;
    beta1_ = beta1;
    beta2_ = beta2;
    epsilon_ = epsilon;

    num_train_ = num_train;
    X_train_ = X_train;
    y_train_ = y_train;
    batch_sz_ = batch_sz;
    eta_ = eta/batch_sz_;
    z_ = vec(M_).fill(0.);


    weights_ = randn<mat>(M_, features_)*(1./sqrt(features_));

    dw_ = mat(M_, features_).fill(0.);
    mom_w_ = mat(M_, features_).fill(0.);
    second_mom_w_ = mat(M_, features_).fill(0.);


    bias_ = randn<vec>(M_);

    db_ = vec(M_).fill(0.);
    mom_b_ = vec(M_).fill(0.);
    second_mom_b_ = vec(M_).fill(0.);

    output_ = vec(M_).fill(0.);

    activation_ = &LogReg::softmax;
    optimizer_ = &LogReg::ADAM;

}


/* Method that runs through epochs and mini-batches to train the model*/
void LogReg::fit(){
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, num_train_-1);
    int idx;
    int batches = num_train_/batch_sz_;
    vec x, y;
    x = vec(features_);
    y = vec(M_);
    for (int epoch = 0; epoch < epochs_; epoch++){
        cout << "epoch = " << epoch << " of " << epochs_ << endl;
        for (batch_ = 0; batch_ < batches; batch_++){
            for (int b = 0; b < batch_sz_; b++){
                idx = distribution(generator);
                for (int j = 0; j < features_; j++){
                    x(j) = X_train_(j, idx);
                }
                for (int j = 0; j < M_; j++){
                    y(j) = y_train_(j, idx);
                }
                z_ = weights_*x + bias_;
                output_ = (this->*activation_)(z_);
                dw_ += compute_dw(x,y) + Lambda_*weights_;
                db_ += compute_db(y);
            }
            (this->*optimizer_)();
        }
    }

}

/* Method for SGD optimization */
void LogReg::SGD()
{
    dw_ *= eta_;
    db_ *= eta_;

    weights_ -= dw_;
    bias_ -= db_;

    dw_.fill(0.);
    db_.fill(0.);

}

/* Method for SGD with momentum optimization */
void LogReg::SGD_momentum()
{
    dw_ *= eta_;
    db_ *= eta_;

    tmp_weights_ = (dw_ + gamma_*tmp_weights_);
    weights_ -= tmp_weights_;

    tmp_bias_ = (db_ + gamma_*tmp_bias_);
    bias_ -= tmp_bias_;

    dw_.fill(0.);
    db_.fill(0.);
}

/* Method for Adam optimization */
void LogReg::ADAM()
{
    dw_ *= eta_;
    db_ *= eta_;

    mom_w_ = beta1_*mom_w_ + (1-beta1_)*dw_;
    mom_b_ = beta1_*mom_b_ + (1-beta1_)*db_;

    second_mom_w_ = beta2_*second_mom_w_ + (1-beta2_)*(dw_%dw_);
    second_mom_b_ = beta2_*second_mom_b_ + (1-beta2_)*(db_%db_);

    alpha_batch_ = eta_*sqrt(1-pow(beta2_, batch_+1))/(1-pow(beta1_, batch_+1));

    epsilon_batch_ = epsilon_*sqrt(1-pow(beta2_, batch_+1));

    weights_ -= mom_w_*alpha_batch_/(sqrt(second_mom_w_) + epsilon_batch_);
    bias_ -= mom_b_*alpha_batch_/(sqrt(second_mom_b_) + epsilon_batch_);

    dw_.fill(0.);
    db_.fill(0.);
}

vec LogReg::softmax(vec a)
{
    vec res = exp(a);
    return exp(a)/sum(res);
}

/* Method for computing the gradient of the cross-entropy with respect to the weights */
mat LogReg::compute_dw(vec x, vec y)
{
    return (output_ - y)*x.t();
}

/* Method for computing the gradient of the cross-entropy with respect to the bias */
vec LogReg::compute_db(vec y)
{
    return output_ - y;
}

/* Method for predicting on unseen data*/
void LogReg::predict(vec x)
{
    z_ = weights_*x + bias_;
    output_ = (this->*activation_)(z_);
}


/* Method for evaluating the score of accuracy when predicting on unseen data */
double LogReg::compute_accuracy(mat X_test, mat y_test, int num_test){

    X_test_ = X_test;
    y_test_ = y_test;
    num_test_ = num_test;

    vec x = vec(features_);
    vec y = vec(M_);

    int idx;
    double accuracy = 0;
    double correct_predictions = 0;
    double wrong_predictions = 0;
    for (int i = 0; i < num_test_; i++){
        for (int j = 0; j < features_; j++){
            x(j) = X_test_(j, i);
        }
        for (int j = 0; j < M_; j++){
            y(j) = y_test_(j, i);
        }
        predict(x);
        idx = index_max(output_);
        if (y(idx) == 1){
            correct_predictions++;
        }
        else{
            wrong_predictions++;
        }
    }
    accuracy = correct_predictions*(1./num_test_);

    cout << "Accuracy = " << accuracy << endl;
    cout << "correct_predictions = " << correct_predictions << endl;
    cout << "wrong_predictions = " << wrong_predictions << endl;

    return accuracy;
}

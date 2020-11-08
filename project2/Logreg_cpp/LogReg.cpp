#include "LogReg.hpp"

LogReg::LogReg(int classes, mat X_train, mat y_train, double eta, double gamma, double Lambda, int epochs, int batch_sz, string optimizer, int num_train, int features)
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


    if (optimizer == "SGD"){
        optimizer_ = &LogReg::SGD;

    }


}

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
        for (int batch = 0; batch < batches; batch++){
            for (b_ = 0; b_ < batch_sz_; b_++){
                idx = distribution(generator);
                for (int j = 0; j < features_; j++){
                    x(j) = X_train_(j, idx);
                }
                for (int j = 0; j < M_; j++){
                    y(j) = y_train_(j, idx);
                }
                z_ = weights_*x + bias_;
                cout <<"z fine"<<endl;
                output_ = (this->*activation_)(z_);
                cout << "output fine" <<endl;
                dw_ += compute_dw(x,y) + Lambda_*weights_;
                cout << "dw fine" << endl;
                db_ += compute_db(y);
                cout << "db fine" << endl;
            }
            (this->*optimizer_)();
        }
    }

}

void LogReg::SGD()
{
    dw_ *= eta_;
    db_ *= eta_;

    weights_ -= dw_;
    bias_ -= db_;

    dw_.fill(0.);
    db_.fill(0.);

}

vec LogReg::softmax(vec a)
{
    vec res = exp(a);
    return exp(a)/sum(res);
}

vec LogReg::compute_dw(vec x, vec y)
{
    return kron((output_ - y),x)
}

vec LogReg::compute_db(vec y)
{
    return output_ - y;
}

#ifndef LOGREG_HPP
#define LOGREG_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <random>
#include <iomanip>
#include <cstdlib>
#include "time.h"
#include <armadillo>

using namespace std;
using namespace arma;


class LogReg {
private:

    double Lambda_, eta_;
    mat X_train_,y_train_, weights_, dw_;
    mat X_test_, y_test_;
    vec bias_, db_, output_, z_;

    int M_, epochs_, batch_sz_, features_, num_train_, num_outputs_, batch_, num_test_;
    //For SGD:
    void SGD();

    //For SGD with momentum;
    void SGD_momentum();
    double gamma_;
    mat tmp_weights_;
    vec tmp_bias_;

    //For ADAM;
    void ADAM();
    double beta1_, beta2_, epsilon_, alpha_batch_, epsilon_batch_;
    mat mom_w_, second_mom_w_, scaled_mom_w_, scaled_second_mom_w_;

    vec mom_b_, second_mom_b_, scaled_mom_b_, scaled_second_mom_b_;

    void (LogReg::*optimizer_)();
    vec (LogReg::*activation_)(vec a);

    vec softmax(vec a);
    mat compute_dw(vec x, vec y);
    vec compute_db(vec y);
    void predict(vec x);




public:

    //Constructor for SGD:
    LogReg(int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features);

    //Constructor for SGD with momentum:
    LogReg(double gamma, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features);

    //Constructor for ADAM:
    LogReg(double beta1, double beta2, double epsilon, int classes, mat X_train, mat y_train, double eta, double Lambda, int epochs, int batch_sz, int num_train, int features);

    double compute_accuracy(mat X_test, mat y_test, int num_test);

    void fit();
};

#endif

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

    double Lambda_, gamma_, eta_;
    mat X_train_,y_train_, weights_, dw_;
    vec bias_, db_, output_, z_;

    int M_, epochs_, batch_sz_, features_, num_train_, num_outputs_, b_;
    void SGD();

    void (LogReg::*optimizer_)();
    vec (LogReg::*activation_)(vec a);

    vec softmax(vec a);
    vec compute_dw(vec x, vec y);
    vec compute_db(vec y);

    //For SGD:


public:

    LogReg(int classes, mat X_train, mat y_train, double eta, double gamma, double Lambda, int epochs, int batch_sz, string optimizer, int num_train, int features);

    void fit();
};

#endif

#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <random>
#include <iomanip>
#include <cstdlib>
#include "time.h"

using namespace std;


class LogisticRegression{
private:
    double Lambda_, gamma_, eta_;
    double *X_data, *y_data;

    int M_, epochs_, batch_sz_;


public:

    LogisticRegression(int classes, double *X_data, double *y_data, double eta, double gamma, double Lambda, int epochs, int batch_sz, string optimizer, int num_points, int features);

};

#endif

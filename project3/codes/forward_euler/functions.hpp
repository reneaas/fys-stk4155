#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <string>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;





void initialize(double **v_new, double **v_old, double **x, int gridpoints, double dx);

void explicit_scheme(double *v_new, double *v_old , double r, int gridpoints, double dt, double total_time, double *t);

void write_to_file(string outfilename, double t, int gridpoints, double *v_new, double *x);


#endif

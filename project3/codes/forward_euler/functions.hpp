#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <string>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;





void initialize(double **v, double **x, double **t, int gridpoints, int timesteps, double dx, double dt);

void explicit_scheme(double *v , double r, int gridpoints, int timesteps);

void write_to_file(string outfilename, double *v, double *x, double *t, int gridpoints, int timesteps);


#endif

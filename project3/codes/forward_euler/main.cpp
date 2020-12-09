#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "time.h"
#include "functions.hpp"

using namespace std;

int main(int nargs, char* argv[]){
    //Declaration of variables.
    double *v, *x, *t;
    int gridpoints;
    double r, dt, dx, total_time;
    double start_x, end_x, timesteps;
    string outfilename;

    outfilename = argv[1];
    total_time = atof(argv[2]);
    dx = atof(argv[3]);


    start_x = 0.;
    end_x = 1.;

    r = 0.5;
    dt = r*dx*dx;
    timesteps = total_time/dt;

    gridpoints = (int) (end_x - start_x)/dx + 1;


    initialize(&v, &x, &t, gridpoints, timesteps, dx, dt);

    explicit_scheme(v, r, gridpoints, timesteps);

    write_to_file(outfilename, v, x, t, gridpoints, timesteps);

    delete[] v;
    delete[] x;
    delete[] t;


    return 0;
}

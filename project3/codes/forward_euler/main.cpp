#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "time.h"
#include "functions.hpp"

using namespace std;

int main(int nargs, char* argv[]){
    //Declaration of variables.
    double *v_new, *v_old, *x;
    int gridpoints;
    double r, t, dt, dx, total_time;
    double start_x, end_x;
    string outfilename;

    outfilename = argv[1];
    total_time = atof(argv[2]);
    dx = atof(argv[3]);


    start_x = 0.;
    end_x = 1.;

    r = 0.5;
    dt = r*dx*dx;
    t = 0;

    gridpoints = (int) (end_x - start_x)/dx + 1;

    initialize(&v_new, &v_old, &x, gridpoints, dx);

    explicit_scheme(v_new, v_old, r, gridpoints, dt, total_time, &t);

    write_to_file(outfilename, t, gridpoints, v_new, x);

    delete[] v_old;
    delete[] v_new;
    delete[] x;


    return 0;
}

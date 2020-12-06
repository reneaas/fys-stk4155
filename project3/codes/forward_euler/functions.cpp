#include "functions.hpp"




void initialize(double **v_new, double **v_old, double **x, int gridpoints, double dx){
    // Position array
    *x = new double[gridpoints];
    for (int i = 0; i < gridpoints; i++) (*x)[i] = dx*(i);


    //Initiate empty solution vector.
    *v_new = new double[gridpoints];
    *v_old = new double[gridpoints];

    //Initial condition
    for (int i = 0; i < gridpoints; i++) (*v_old)[i] = sin(M_PI * (*x)[i]);

}


//Algorithm for forward Euler method
void explicit_scheme(double *v_new, double *v_old , double r, int gridpoints, double dt, double total_time, double *t){

    while (*t < total_time){
        for (int j = 1; j < gridpoints-1; j++){
            v_new[j] = (1-2*r)*v_old[j] + r*(v_old[j+1] + v_old[j-1]);
        }
        for (int k = 0; k < gridpoints; k++) v_old[k] = v_new[k];
        *t += dt;
    }
}


void write_to_file(string outfilename, double t, int gridpoints, double *v_new, double *x){
    ofstream ofile;
    cout << "Writing to file for t = " << t << endl;
    ofile.open(outfilename);
    ofile << t << endl;
    for (int i = 0; i < gridpoints; i++){
        ofile << x[i] << " " << v_new[i] << endl;
    }
    ofile.close();

}

#include "functions.hpp"




void initialize(double **v, double **x, double **t, int gridpoints, int timesteps, double dx, double dt){
    // Position array
    *x = new double[gridpoints]();
    for (int i = 0; i < gridpoints; i++) (*x)[i] = dx*i;

    // Time array
    *t = new double[timesteps]();
    for (int i = 0; i < timesteps; i++) (*t)[i] = dt*i;


    //Initialize solution vector.
    int tot_points = gridpoints*timesteps;
    *v = new double[tot_points]();


    //Initial condition
    for (int i = 0; i < gridpoints; i++) (*v)[i] = sin(M_PI * (*x)[i]);

}


//Algorithm for forward Euler scheme
void explicit_scheme(double *v, double r, int gridpoints, int timesteps){

    for (int i = 0; i < timesteps; i++){
        for (int j = 1; j < gridpoints-1; j++){
            v[(i+1)*gridpoints + j] = (1-2*r)*v[i*gridpoints + j] + r*(v[i*gridpoints + (j+1)] + v[i*gridpoints + (j-1)]);
        }
    }
}


void write_to_file(string outfilename, double *v, double *x, double *t, int gridpoints, int timesteps){
    ofstream ofile;
    ofile.open(outfilename);
    ofile << timesteps << " " << gridpoints-2 << endl;

    for (int i = 0; i < timesteps; i++){
        for (int j = 1; j < gridpoints-1; j++){
            ofile << t[i] << " " << x[j] << " " << v[i*gridpoints + j] << endl;
        }
    }
    ofile.close();
}

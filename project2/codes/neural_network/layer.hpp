#ifndef LAYER_HPP
#define LAYER_HPP

#include <armadillo>

using namespace std;
using namespace arma;


/*
 The Layer class is used by the FFNN class to store properties of each layer.
*/
class Layer {
public:

    int rows_, cols_; //Number of rows and columns in the weights_ matrix.

    //Properties a layer in the neural net.
    vec bias_;
    vec error_;
    vec activation_;
    mat weights_;
    vec z_;

    //Store gradients
    mat dw_;
    vec db_;

    //Used with sgd_momentum
    mat w_mom_;
    vec b_mom_;


    //Constructors
    Layer(int rows, int cols);
    Layer(int rows, int cols, string optimizer);
};

#endif

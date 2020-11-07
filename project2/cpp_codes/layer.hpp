#ifndef LAYER_HPP
#define LAYER_HPP

#include <armadillo>

using namespace std;
using namespace arma;

class Layer {
public:
    /* data */

    vec bias_;
    vec error_;
    vec activation_;
    mat weights_;
    vec z_;

    //Store gradients
    mat dw_;
    vec db_;

    //Momentum parameters
    mat w_mom_;
    vec b_mom_;

    int rows_, cols_;

    //Constructions
    Layer(int rows, int cols);
    Layer(int rows, int cols, string optimizer);
};

#endif

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

    int rows_, cols_;

    Layer(int rows, int cols);
    void compute_act(vec a);
};

#endif

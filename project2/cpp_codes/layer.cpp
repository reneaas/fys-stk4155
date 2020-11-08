#include "layer.hpp"

Layer::Layer(int rows, int cols)
{
    rows_ = rows;
    cols_ = cols;
    weights_ = randn<mat>(rows_, cols_)*(1./sqrt(cols_));
    bias_ = randn<vec>(rows_);
    activation_ = vec(rows_).fill(0.);
    z_ = vec(rows_).fill(0.);
    error_ = vec(rows_).fill(0.);

    //Initialize gradients
    dw_ = mat(rows_, cols_).fill(0.);
    db_ = vec(rows_).fill(0.);
}

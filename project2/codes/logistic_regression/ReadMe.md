# Logistic Regression

This folder contains code produced for a Logistic Regression model. The model implemented has the ability to be trained using three optimizers:

* Stochastic Gradient Descent (SGD)
* Stochastic Gradient Descent with momentum
* Adam

### Using the model
The implemented model class in *LogReg.cpp* has three different constructors which correspond to the three methods of optimization. The file *main.cpp* provides a simple example of how to train the model and perform an analysis of accuracy in predictions when tested on a test set using the mnist dataset. This test run uses a set of predetermined hyperparameters defined in the main block but are easily changed if one wishes to look at results for different values.



This is easily tested by using the included *makefile* which can be called in your terminal window by typing

```console
make all
```

or by compiling and running from terminal using

```console
c++ -o main.out LogReg.cpp main.cpp -larmadillo -Ofast -std=c++11
./main.out
```

The resulting output should be something like

```console
Training model using SGD...
Testing model...
Accuracy = 0.9218
correct_predictions = 9218
wrong_predictions = 782

Training model using SGD w/momentum...
Testing model...
Accuracy = 0.9215
correct_predictions = 9215
wrong_predictions = 785

Training model using ADAM...
Testing model...
Accuracy = 0.9214
correct_predictions = 9214
wrong_predictions = 786
```

from LogisticRegression import LogReg
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data_valid, predict_model_mnist, design_matrix, plot_mnist_weights
import os

Ntrain = 55000
Ntest = 10000
Nvalid = 5000
Xtrain, Ytrain, Xtest, Ytest, Xvalid, Yvalid = mnist_data_valid(Ntrain, Ntest, Nvalid)


def train_and_test_mnist(N_train, N_test, X_train, Y_train, X_test, Y_test, classes, eta, gamma, Lambda, epochs, batch_size, optimizer):
    my_solver = LogReg(classes= classes, X_data = X_train, y_data = Y_train, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs, batch_size=batch_size, optimizer = optimizer)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = predict_model_mnist(my_solver, X_test, Y_test, N_test)


    # plot weights vs the pixel position
    weights = my_solver.weights.copy()
    plot_mnist_weights(weights, eta, gamma, epochs, batch_size, Lambda, accuracy)


#LA STÅ FUCKERS, HAR TWEAKA
train_and_test_mnist(N_train=Ntrain, N_test=Ntest, X_train = Xtrain, Y_train = Ytrain, X_test = Xtest, Y_test = Ytest, classes = 10, eta = 10**(-3), gamma = 0.1, epochs=5, Lambda = 10**(-8), batch_size = 500, optimizer = "ADAM")


"""
Eta = [0.01*i for i in range(1,11)]
Gamma = [(0.15 + 0.01*i) for i in range(0,11)]
Accuracy = []
Lambda = 0.00001
epochs = 10
classes = 10
batch_size = 100
for eta in Eta:
    for gamma in Gamma:
        my_solver = LogReg(classes= classes, X_data = Xtrain, y_data = Ytrain, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs, batch_size=batch_size)
        my_solver.train()
        accuracy = predict_model_mnist(my_solver, Xvalid, Yvalid, Nvalid)
        Accuracy.append(accuracy)

Eta = np.array(Eta)
Gamma = np.array(Gamma)
Accuracy = np.array(Accuracy)

path = "./results/LogisticRegression/"
if not os.path.exists(path):
    os.makedirs(path)
filename_eta = path + "LogReg_Eta.npy"
filename_gamma = path + "LogReg_Gamma.npy"
filename_accuracy = path + "LogReg_Accuracy.npy"

np.save(filename_eta, Eta)
np.save(filename_gamma, Gamma)
np.save(filename_accuracy, Accuracy)

"""
"""
Eta = [10**(-i/2) for i in range(1,11)]
Gamma = [0.1*i for i in range(0,10)]
Accuracy = []
Lambda = 0.00001
epochs = 10
classes = 10
batch_size = 100
for eta in Eta:
    for gamma in Gamma:
        my_solver = LogReg(classes= classes, X_data = Xtrain, y_data = Ytrain, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs, batch_size=batch_size)
        my_solver.train()
        accuracy = predict_model_mnist(my_solver, Xvalid, Yvalid, Nvalid)
        Accuracy.append(accuracy)

Eta = np.array(Eta)
Gamma = np.array(Gamma)
Accuracy = np.array(Accuracy)

path = "./results/LogisticRegression/"
if not os.path.exists(path):
    os.makedirs(path)
filename_eta = path + "LogReg_Eta_broad.npy"
filename_gamma = path + "LogReg_Gamma_broad.npy"
filename_accuracy = path +"LogReg_Accuracy_broad.npy"

np.save(filename_eta, Eta)
np.save(filename_gamma, Gamma)
np.save(filename_accuracy, Accuracy)
"""
"""
eta = 10**(-3/2)
gamma = 0.1
Lambda = [10**(i) for i in range(-9,0)]
print(Lambda)
Accuracy = []
epochs = 10
classes = 10
batch_size = 100
for lamda in Lambda:
    my_solver = LogReg(classes= classes, X_data = Xtrain, y_data = Ytrain, eta = eta, gamma = gamma, Lambda = lamda, epochs = epochs, batch_size=batch_size)
    my_solver.train()
    accuracy = predict_model_mnist(my_solver, Xvalid, Yvalid, Nvalid)
    Accuracy.append(accuracy)

Lambda = np.array(Lambda)
Accuracy = np.array(Accuracy)

path = "./results/LogisticRegression/"
if not os.path.exists(path):
    os.makedirs(path)
filename_lambda = path + "LogReg_lambda.npy"
filename_accuracy = path +"LogReg_Accuracy_for_Lambda.npy"

np.save(filename_lambda, Lambda)
np.save(filename_accuracy, Accuracy)
"""

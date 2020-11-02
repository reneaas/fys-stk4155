from LogisticRegression import LogReg
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, predict_model_mnist, design_matrix
import os

Ntrain = 60000
Ntest = 10000
Xtrain, Ytrain, Xtest, Ytest = mnist_data(Ntrain, Ntest)

def train_and_test_mnist(N_train, N_test, X_train, Y_train, X_test, Y_test, classes, eta, gamma, Lambda, epochs, batch_size):
    my_solver = LogReg(classes= classes, X_data = X_train, y_data = Y_train, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs, batch_size=batch_size)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = predict_model_mnist(my_solver, X_test, Y_test, N_test)


    # plot weights vs the pixel position
    weights = my_solver.weights.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(weights).max()
    for i in range(10):
        l2_plot = plt.subplot(2, 5, i + 1)
        l2_plot.imshow(weights[i].reshape(28, 28), interpolation='nearest',cmap=plt.cm.Greys)
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())
        l2_plot.set_xlabel('Class %i' % i)
    plt.suptitle('classification weights vector $w_j$ for digit class $j$\n eta = %f , gamma = %f , ephocs = %i, batch_size = %i, lambda = %f, accuracy = %f %%' %(eta, gamma, epochs, batch_size, Lambda, accuracy*100))

    plt.show()

#LA STÅ FUCKERS, HAR TWEAKA
train_and_test_mnist(N_train=Ntrain, N_test=Ntest, X_train = Xtrain, Y_train = Ytrain, X_test = Xtest, Y_test = Ytest, classes = 10, eta = 0.01, gamma = 0.22, epochs=10, Lambda = 0.00001, batch_size = 100)
"""
Eta = [0.01*i for i in range(0,11)]
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
        accuracy = predict_model_mnist(my_solver, Xtest, Ytest, Ntest)
        Accuracy.append(accuracy)

Eta = np.array(Eta)
Gamma = np.array(Gamma)
Accuracy = np.array(Accuracy)

path = "./results/"
if not os.path.exists(path):
    os.makedirs(path)
filename_eta = "LogReg_Eta.npy"
filename_gamma = "LogReg_Gamma.npy"
filename_accuracy = "LogReg_Accuracy.npy"

np.save(filename_eta, Eta)
np.save(filename_gamma, Gamma)
np.save(filename_accuracy, Accuracy)
"""

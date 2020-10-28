from LogisticRegression import LogReg
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, test_model_mnist_single, design_matrix

Ntrain = 50000
Ntest = 10000
def train_and_test_mnist(N_train, N_test, classes, eta, gamma, Lambda, epochs, dig):
    X_train, Y_train, X_test, Y_test = mnist_data(N_train, N_test)


    yy_test = np.zeros(N_test)
    yy_train = np.zeros(N_train)


    for i in range(N_train):
        max_where = np.where(Y_train[i] == 1)[0][0]
        if max_where == dig:
            yy_train[i] = 1

    for i in range(N_test):
        max_where = np.where(Y_test[i] == 1)[0][0]
        if max_where == dig:
            yy_test[i] = 1


    my_solver = LogReg(classes= classes, X_data = X_train, y_data = yy_train, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = test_model_mnist_single(my_solver, X_test, yy_test, N_test)
    print("Accuracy = ", accuracy)


    # plot weights vs the pixel position
    weights = my_solver.weights.copy()
    plt.figure()
    scale = np.abs(weights).max()
    l2_plot = plt.subplot(2, 1, 1)
    l2_plot.imshow(weights.reshape(28, 28), interpolation='nearest',cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())
    l2_plot.set_xlabel('Class 5')
    plt.suptitle('classification weights vector $w_j$ for digit class $j$')

    plt.show()


train_and_test_mnist(N_train=Ntrain, N_test=Ntest, classes = 1, eta = 0.1, gamma = 0.9, epochs=10, Lambda = 0.00001, dig = 2)

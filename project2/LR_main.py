from LogisticRegression import LogReg
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, test_model_mnist, design_matrix

Ntrain = 60000
Ntest = 10000
def train_and_test_mnist(N_train, N_test, classes, eta, gamma, Lambda, epochs):
    X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)
    my_solver = LogReg(classes= classes, X_data = X_train, y_data = Y_train, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = test_model_mnist(my_solver, X_test, Y_test, Ntest)
    print("Accuracy = ", accuracy)

    w_mat_0 = np.zeros([28,28])
    w_mat_0.flat[:] = my_solver.weights[1,:]

    """
    x = np.linspace(1,28,28)
    y = np.linspace(1,28,28)
    x,y = np.meshgrid(x,y)
    plt.contourf(x,y,w_mat_0)
    plt.show
    """

    # plot weights vs the pixel position
    weights = my_solver.weights.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(weights).max()
    for i in range(10):
        l2_plot = plt.subplot(2, 5, i + 1)
        l2_plot.imshow(weights[i].reshape(28, 28), interpolation='nearest',
                       cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())
        l2_plot.set_xlabel('Class %i' % i)
    plt.suptitle('classification weights vector $w_j$ for digit class $j$')

    plt.show()


train_and_test_mnist(N_train=Ntrain, N_test=Ntest, classes = 10, eta = 0.1, gamma = 0.9, epochs=10, Lambda = 0.00001)

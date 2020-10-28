from LogisticRegression import LogReg
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, test_model_mnist, design_matrix

Ntrain = 60000
Ntest = 10000
def train_and_test_mnist(N_train, N_test, classes, eta, gamma, Lambda, epochs, batch_size):
    X_train, Y_train, X_test, Y_test = mnist_data(N_train, N_test)
    my_solver = LogReg(classes= classes, X_data = X_train, y_data = Y_train, eta = eta, gamma = gamma, Lambda = Lambda, epochs = epochs, batch_size=batch_size)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = test_model_mnist(my_solver, X_test, Y_test, N_test)
    print("Accuracy = ", accuracy)


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
    plt.suptitle('classification weights vector $w_j$ for digit class $j$\n eta = %f , gamma = %f , ephocs = %i, batch_size = %i, lambda = %f, accuracy = %f %%' %(eta, gamma, epochs, batch_size, Lambda, accuracy*100))

    plt.show()

#LA STÅ FUCKERS, HAR TWEAKA
train_and_test_mnist(N_train=Ntrain, N_test=Ntest, classes = 10, eta = 0.01, gamma = 0.22, epochs=10, Lambda = 0.00001, batch_size = 100)

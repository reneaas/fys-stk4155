from NeuralNetwork import FFNN
import numpy as np
import tensorflow as tf
from time import time

def scale_data(X, y, Npoints):
    if y == None:
        for i in range(Npoints):
            x_mean = np.mean(X[i])
            x_std = np.std(X[i])
            X[i] = (X[i]-x_mean)/x_std

    else:
        for i in range(Npoints):
            x_mean = np.mean(X[i])
            x_std = np.std(X[i])
            X[i] = (X[i]-x_mean)/x_std
            y_mean = np.mean(y[i])
            y_std = np.mean(y[i])
            y[i] = (y[i]-y_mean)/y_std


    return X, y

def mnist_data(Ntrain, Ntest):
    print("Loading dataset...\n")
    mnist = tf.keras.datasets.mnist
    (trainX, trainY), (testX, testY) = mnist.load_data()
    N_points_train, n, m = np.shape(trainX)
    shuffled_indices = np.random.permutation(N_points_train)
    trainX, trainY = trainX[shuffled_indices], trainY[shuffled_indices]
    N_points_train, n, m = np.shape(testX)
    shuffled_indices = np.random.permutation(N_points_train)
    testX, testY = testX[shuffled_indices], testY[shuffled_indices]


    #Prepare training data
    trainX = trainX/255.0
    X_train = np.zeros((Ntrain, n*m))
    y_values = np.arange(0,10)
    Y_train = np.zeros((Ntrain, 10))
    for i in range(Ntrain):
        X_train[i] = trainX[i].flat[:]
        Y_train[i] = trainY[i] == y_values

    scale_data(X = X_train, y = None, Npoints = Ntrain)

    #Prepare test data
    testX = testX/255.0
    X_test = np.zeros((Ntest, n*m))
    Y_test = np.zeros((Ntest, 10))

    for i in range(Ntest):
        X_test[i] = testX[i].flat[:]
        Y_test[i] = testY[i] == y_values

    scale_data(X = X_test, y = None, Npoints = Ntest)

    return X_train, Y_train, X_test, Y_test


def test_model_mnist(my_solver, X_test, Y_test, Ntests):
    total_images = 0
    correct_predictions = 0
    for i in range(Ntests):
        y_predict = my_solver.predict(X_test[i])
        one_hot_prediction = 1.*(y_predict == np.max(y_predict))
        idx = np.where(one_hot_prediction == 1.)
        correct_predictions += (one_hot_prediction[idx] == Y_test[i][idx])
        total_images += 1

    accuracy = correct_predictions/total_images
    return accuracy


#Set up data:
Ntrain = 5000
Ntest = 500
X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)

print(np.shape(X_train))


my_solver = FFNN(layers = 5, nodes = 100, X_data = X_train, y_data = Y_train, N_outputs = 10, hidden_activation = "sigmoid", epochs = 30)

start = time()
my_solver.train()
end = time()
timeused = end - start
print("Timeused = ", timeused)
accuracy = test_model_mnist(my_solver, X_test, Y_test, Ntest)
print("Accuracy = ", accuracy)

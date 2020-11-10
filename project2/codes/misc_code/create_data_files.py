import numpy as np
import tensorflow as tf
import sys
import os
np.random.seed(1001)


def scale_data(X, y, Npoints):
    if not isinstance(y, np.ndarray):
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
    mnist = tf.keras.datasets.mnist
    (trainX, trainY), (testX, testY) = mnist.load_data()
    N_points_train, n, m = np.shape(trainX)
    shuffled_indices = np.random.permutation(N_points_train)
    trainX, trainY = trainX[shuffled_indices], trainY[shuffled_indices]
    N_points_test, n, m = np.shape(testX)
    shuffled_indices = np.random.permutation(N_points_test)
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


N = 20000
sigma = 0.1

x = np.random.uniform(0,1, N)
y = np.random.uniform(0,1, N)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

noise = np.random.normal(0,sigma,size=N)

data = FrankeFunction(x,y) + noise

data = (data - np.mean(data))/np.std(data)
x = (x - np.mean(x))/np.std(x)
y = (y - np.mean(y))/np.std(y)


names = ["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)  ]
outfilename = "_".join(names) + ".txt"

with open(outfilename, "w") as outfile:
    for i in range(N):
        outfile.write(str(x[i]) + " " + str(y[i]) + " " + str(data[i]))
        outfile.write("\n")




Ntrain = 60000
Ntest = 10000
features=28*28
outputs = 10
X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)
outfilename_train_X = "mnist_training_X.txt"
outfilename_train_Y = "mnist_training_Y.txt"
outfilename_test_X = "mnist_test_X.txt"
outfilename_test_Y = "mnist_test_Y.txt"

with open(outfilename_train_X, "w") as outfile:
    for i in range(len(X_train.flat)):
        outfile.write(str(X_train.flat[i]))
        outfile.write("\n")

with open(outfilename_train_Y, "w") as outfile:
    for i in range(Ntrain*outputs):
        outfile.write(str(Y_train.flat[i]))
        outfile.write("\n")


with open(outfilename_test_X, "w") as outfile:
    for i in range(Ntest*features):
        outfile.write(str(X_test.flat[i]))
        outfile.write("\n")

with open(outfilename_test_Y, "w") as outfile:
    for i in range(Ntest*outputs):
        outfile.write(str(Y_test.flat[i]))
        outfile.write("\n")

os.system("g++ -o create_datasets.out create_datasets.cpp -larmadillo")
os.system("./create_datasets.out")

path = ""
os.system("mv")
os.system("rm *.txt")

os.system("mv *.bin ../data_files")

from NeuralNetwork2 import FFNN
import numpy as np
import tensorflow as tf

print("Loading dataset...")
mnist = tf.keras.datasets.mnist
(trainX, trainY), (testX, testY) = mnist.load_data()
Ntrain, n, m = np.shape(trainX)
shuffled_indices = np.random.permutation(Ntrain)
trainX, trainY = trainX[shuffled_indices], trainY[shuffled_indices]
Ntest, n, m = np.shape(testX)
shuffled_indices = np.random.permutation(Ntest)
testX, testY = testX[shuffled_indices], testY[shuffled_indices]

print(np.shape(trainY))

print("Flatten and scale dataset...")
size_of_dataset = 10000
trainX = trainX/255.0
trainX_flat = np.zeros((size_of_dataset, n*m))


#Flatten and scale data

y_training = np.zeros([size_of_dataset, 10])
y_values = np.arange(0,10)

for k in range(size_of_dataset):
    y_training[k] = trainY[k] == y_values
    x = trainX[k].flat[:]
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x - x_mean)/x_std
    trainX_flat[k] = x[:]



#Test data
n_tests = 10
X_test = np.zeros([n_tests, n*m])
y_test = np.zeros([n_tests, 10])
for i in range(n_tests):
    x = testX[i].flat[:]
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x - x_mean)/x_std

    X_test[i] = x
    y_test[i] = testY[i] == y_values


my_solver = FFNN(layers = 5, nodes = 100, X_data = trainX_flat, y_data = y_training, M_outputs = 10)
my_solver.train()
for i in range(n_tests):
    print("Image = ", i)
    y_predict = my_solver.predict(X_test[i])
    #print("Prediction ", y_predict == np.max(y_predict))
    print("Prediction = ", y_predict)
    print("Ground truth = ", y_test[i])

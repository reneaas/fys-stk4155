import keras, sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from functions import mnist_data_valid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
np.random.seed(1001)

Ntrain = 55000
Ntest = 10000
Nvalid = 5000
X_train, Y_train, X_test, Y_test, X_valid, Y_valid = mnist_data_valid(Ntrain, Ntest, Nvalid)

classes = 10
size = 28

#Create model architecture
def create_DNN(val):
    # instantiate model
    model = Sequential()
    # add a dense all-to-all sigmoid layer
    model.add(Dense(400,input_shape=(size*size,), activation='sigmoid', kernel_regularizer=l2(val)))
    # add a dense all-to-all sigmoid layer
    model.add(Dense(100, activation='sigmoid'))
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # soft-max layer
    model.add(Dense(classes, activation='softmax'))

    return model

#Compile model to be trained
def compile_model(learn_rate = 0.01, momentum = 0, val = 0.1):
    # create the model
    model=create_DNN(val)
    # compile the model
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# training parameters
batch_size = 100
epochs = 10

values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
all_test = list()
for param in values:
    # define model
    model_DNN=compile_model(learn_rate = 10**(-3/2), momentum = 0.1, val = param)
    # fit model
    model_DNN.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
    # evaluate the model
    test_acc =  model_DNN.evaluate(X_test, Y_test, verbose=1)[1]
    print("Param: %f, Test: %.3f" % (param, test_acc))
    all_test.append(test_acc)

# plot train and test means
plt.semilogx(values, all_test, label='test', marker='o')
plt.legend()
plt.savefig("./results/Keras_mnist_lambda.pdf")
plt.show()

"""
# create the deep neural net
model_DNN=compile_model(learn_rate =, momentum = )

# train DNN and store training info in history
history=model_DNN.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

# evaluate model
score = model_DNN.evaluate(X_test, Y_test, verbose=1)

# print performance
print()
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# look into training history

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()
"""


"""
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

learn_rate = [10**(-3/2)]
momentum = [0.1]

# call Keras scikit wrapper
model_gridsearch = KerasClassifier(build_fn=compile_model,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)


# define parameter dictionary
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
# call scikit grid search module
grid = GridSearchCV(estimator=model_gridsearch, param_grid=param_grid, n_jobs=1, cv=4)
grid_result = grid.fit(X_train,Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#grid search which resulted in optimal eta = , gamma =
"""

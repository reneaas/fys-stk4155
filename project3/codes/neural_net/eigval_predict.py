from neural_eigensolver import NeuralEigenSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


model_path = "saved_model/eigenvalue_model"

my_model = tf.keras.models.load_model(model_path)

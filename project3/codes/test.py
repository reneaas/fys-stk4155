import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_addons as tfa



a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[3, 3])

shape = a.get_shape().as_list()

b = tf.eye(shape[0])

print(b)

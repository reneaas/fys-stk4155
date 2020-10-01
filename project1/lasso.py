from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from ridge import Ridge

class Lasso(Ridge):

    def __init__(self, Lambda = None):
        super().__init__(Lambda)
        self.Lambda = Lambda

    def train(self):
        self.X_train = np.asfortranarray(self.X_train)
        self.f_train = np.asfortranarray(self.f_train)
        self.clf_lasso = linear_model.Lasso(alpha=self.Lambda, max_iter=1000, normalize=False).fit(self.X_train, self.f_train)
        self.w = (self.clf_lasso.coef_)
        self.w[0] = float(self.clf_lasso.intercept_)

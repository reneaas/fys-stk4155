from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from ridge import Ridge

class Lasso(Ridge):

    def __init__(self, Lambda):
        super().__init__(Lambda)
        self.Lambda = Lambda

    def train(self, X_train, y_train):
        self.clf_lasso = linear_model.Lasso(alpha=self.Lambda).fit(X_train, y_train)
        self.w = (self.clf_lasso.coef_)
        self.w[0] = float(self.clf_lasso.intercept_)

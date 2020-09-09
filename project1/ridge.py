

class Ridge(OLS):
    def __init__(self, Lambda):
        super().__init__()
        self.lambda = Lambda

    def train(self):
        """
        Perform Ridge to find the parameters of the model denoted w.
        """
        A = self.X_train.T @ self.X_train
        shape = np.shape(A)
        A += self.eye(shape[0])
        b = self.X_train.T @ self.y_train
        self.w = np.linalg.solve(A, b)

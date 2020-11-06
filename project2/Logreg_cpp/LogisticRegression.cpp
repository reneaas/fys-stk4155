#include "LogisticRegression.hpp"

LogisticRegression::LogisticRegression(int classes, double *X_data, double *y_data, double eta, double gamma, double Lambda, int epochs, int batch_sz, string optimizer, int num_points, int features){

    M_ = classes;
    epochs_ = epochs;

    X_data_ = new double[]

    X_data_ = X_data;
    y_data_ = y_data;
    self.N_points, self.features = np.shape(X_data)

    self.Lambda = Lambda
    self.batch_size = batch_size
    self.eta = eta/self.batch_size

    self.grad_weights = np.zeros([self.M,self.features])
    self.weights = np.random.normal(size=[self.M,self.features])/np.sqrt(self.features)

    self.bias = np.random.uniform(size=self.M)
    self.grad_bias = np.zeros(self.M)

    self.output = np.zeros(self.M)
}


import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto'):
        self.C = C  # Regularization parameter
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None

    def _linear_kernel(self, X, Y):
        return np.dot(X, Y.T)

    def _calculate_kernel_matrix(self, X):
        if self.kernel == 'linear':
            return self._linear_kernel(X, X)
        # Add other kernel functions like polynomial or RBF here

    def _objective(self, alpha):
        return 0.5 * np.dot(alpha, np.dot(self.kernel_matrix_, alpha)) - np.sum(alpha)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.kernel_matrix_ = self._calculate_kernel_matrix(X)
        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y)}
        alpha_init = np.zeros(n_samples)
        res = minimize(self._objective, alpha_init, bounds=bounds, constraints=constraints)
        self.dual_coef_ = res.x
        support_vector_indices = np.where((self.dual_coef_ > 1e-5) & (self.C - self.dual_coef_ > 1e-5))[0]
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_labels_ = y[support_vector_indices]
        self.intercept_ = np.mean(self.support_vector_labels_ - np.dot(self.kernel_matrix_[support_vector_indices], self.dual_coef_))

    def predict(self, X):
        kernel_values = self._linear_kernel(X, self.support_vectors_)
        return np.sign(np.dot(kernel_values, self.dual_coef_) + self.intercept_)

import numpy as np


== ord norm for matrices norm for vectors =============================================================== None Frobenius norm 2-norm 'fro' Frobenius norm -- 'nuc' nuclear norm -- inf max(sum(abs(x), axis=1)) max(abs(x)) -inf min(sum(abs(x), axis=1)) min(abs(x)) 0 -- sum(x != 0) 1 max(sum(abs(x), axis=0)) as below -1 min(sum(abs(x), axis=0)) as below 2 2-norm (largest sing. value) as below -2 smallest singular value as below other -- sum(abs(x)**ord)**(1./ord) ===============================================================

The Frobenius norm is given by [1]_:

    ||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}


# ||W|| = sqrt([\sum_{i,j} abs(a_{i,j})^2])

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-5):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def _gradient(self, func, x):
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            x_plus_h = x.copy()
            x_plus_h[i] += self.tolerance
            gradient[i] = (func(x_plus_h) - func(x)) / self.tolerance
        return gradient

    def minimize(self, func, x_init):
        x = x_init
        for _ in range(self.max_iterations):
            gradient = self._gradient(func, x)
            x_new = x - self.learning_rate * gradient
            if np.linalg.norm(x_new - x) < self.tolerance:
                break
            x = x_new
        return x

# Example usage:
def quadratic_function(x):
    return x[0]**2 + x[1]**2

gd = GradientDescent(learning_rate=0.1)
x_init = np.array([3.0, 4.0])
minimum = gd.minimize(quadratic_function, x_init)
print("Minimum:", minimum)
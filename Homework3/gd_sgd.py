import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# f can be three different functions
# first one -> f(x1,x2) = (x1-3)**2 + (x2-1)**2
# second one -> f(x1,x2) = 10(x1-1)**2 + (x2-2)**2
# third one -> f(x) = 0.5*np.linalg.norm(Ax-b)**2 where A is the vandermonde matrix and x_true is a vector of ones
# fifth one -> f(x) = 0.5*np.linalg.norm(Ax-b)**2 + 0.5*c*np.linalg.norm(x)**2 where A is the vandermonde matrix
# and x_true is a vector of ones and c is a parameter that we can change in the range [0, 1]
def f(x):
    # first function
    # return (x[0] - 3) ** 2 + (x[1] - 1) ** 2
    # fifth function
    x_true = np.ones(10)
    A = np.vander(np.linspace(0, 1, 10), 10, increasing=True)
    b = A @ x_true
    c = 0.5
    return 0.5 * np.linalg.norm(A @ x - b) ** 2 + 0.5 * c * np.linalg.norm(x) ** 2


def grad_f(x):
    # first function
    # return np.array([2 * (x[0] - 3), 2 * (x[1] - 1)])
    # fifth function
    x_true = np.ones(10)
    A = np.vander(np.linspace(0, 1, 10), 10, increasing=True)
    b = A @ x_true
    c = 0.5
    return A.T @ (A @ x - b) + c * x


def backtracking(f, grad_f, x):
    """
    This function is a simple implementation of the backtracking algorithm for
    the GD (Gradient Descent) method.

    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    x: ndarray. The actual iterate x_k.
    """
    alpha = 1
    c = 0.8
    tau = 0.25

    while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.linalg.norm(grad_f(x), 2) ** 2:
        alpha = tau * alpha

        if alpha < 1e-3:
            break
    return alpha


# implementation of the gradient descent algorithm
# input:
# f: function to be minimized
# grad_f: gradient of f, supposed to be a function
# x0: an n-dimensional vector, initial point
# kmax: maximum number of iterations
# tolx: tolerance on the norm of the gradient
# tolf: tolerance on the value of f
# output:
# x: an array that contains the value of x_k for each iterate x_k
# k: number of iterations
# f_val: an array that contains the value of f(x_k) for each iterate x_k
# grads: an array that contains the value of the gradient at x_k for each iterate x_k
# err: an array that contains the value of the norm of the gradient at x_k for each iterate x_k
def gradient_descent(f, grad_f, x0, kmax, tolx, tolf, back=False):
    x = [x0]
    f_val = [f(x0)]
    grads = [grad_f(x0)]
    err = [np.linalg.norm(grad_f(x0), 2)]
    k = 0
    while k < kmax and err[k] > tolx and f_val[k] > tolf:
        if back:
            alpha = backtracking(f, grad_f, x[k])
        else:
            alpha = 0.01
        x.append(x[k] - alpha * grad_f(x[k]))
        f_val.append(f(x[k + 1]))
        grads.append(grad_f(x[k + 1]))
        err.append(np.linalg.norm(grad_f(x[k + 1]), 2))
        k += 1
    return x, k, f_val, grads, err


# compute the gradient descent for the first function
# x0 = np.array([0, 0])
# x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, 1000, 1e-6, 1e-6, True)
# # plot the results
# plt.figure()
# plt.plot(np.arange(k + 1), f_val)
# plt.xlabel('k')
# plt.ylabel('f(x_k)')
# plt.title('f(x_k) for the first function')
# plt.savefig('f(x_k)_first_function.png')
# plt.clf()
# # compute the gradient descent for the first function without backtracking
# x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, 1000, 1e-6, 1e-6, False)
# # plot the results
# plt.figure()
# plt.plot(np.arange(k + 1), f_val)
# plt.xlabel('k')
# plt.ylabel('f(x_k)')
# plt.title('f(x_k) for the first function without backtracking')
# plt.savefig('f(x_k)_first_function_without_backtracking.png')
# plt.clf()
# # plot the contour of the first function around the minimum and the path of the gradient descent
# x1 = np.linspace(-1, 5, 100)
# x2 = np.linspace(-1, 5, 100)
# X1, X2 = np.meshgrid(x1, x2)
# Z = (X1 - 3) ** 2 + (X2 - 1) ** 2
# plt.figure()
# plt.contour(X1, X2, Z, 20)
# plt.plot([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))], 'r')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Contour of the first function and path of the gradient descent')
# plt.savefig('contour_first_function.png')
# plt.clf()
# # plot the contour of the first function around the minimum and the path of the gradient descent without backtracking
# Z = (X1 - 3) ** 2 + (X2 - 1) ** 2
# plt.figure()
# plt.contour(X1, X2, Z, 20)
# plt.plot([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))], 'r')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Contour of the first function and path of the gradient descent without backtracking')
# plt.savefig('contour_first_function_without_backtracking.png')
# plt.clf()

# compute the gradient descent for the fifth function
x0 = np.zeros(10)
x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, 1000, 1e-6, 1e-6, True)
# plot the results
plt.figure()
plt.plot(np.arange(k + 1), f_val)
plt.xlabel('k')
plt.ylabel('f(x_k)')
plt.title('f(x_k) for the fifth function')
plt.savefig('f(x_k)_fifth_function.png')
plt.clf()
# compute the gradient descent for the fifth function without backtracking
x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, 1000, 1e-6, 1e-6, False)
# plot the results
plt.figure()
plt.plot(np.arange(k + 1), f_val)
plt.xlabel('k')
plt.ylabel('f(x_k)')
plt.title('f(x_k) for the fifth function without backtracking')
plt.savefig('f(x_k)_fifth_function_without_backtracking.png')
plt.clf()


# function l(w, D):
def l(w, D):
    return np.sum(np.log(1 + np.exp(-D * np.dot(D, w))))


# implementation of the stochastic gradient descent algorithm
# input:
# l: (w, D) function to be minimized where D is the dataset and w is the weight vector
# grad_l: gradient of l, supposed to be a function
# w0: an n-dimensional array that contains the initial iterate. By default, it should be randomly sampled
# data: a tuple (x, y) that contains two arrays, where x is the input data and y is the output data
# batch_size: size of the batch, should be a divisor of the number of data points
# n_epochs: number of epochs
# output:
# w: an array that contains the value of the weight vector at w_k for each iterate w_k
# f_val: an array that contains the value of the function at w_k for each iterate w_k only after each epoch
# grads: an array that contains the value of the gradient at w_k for each iterate w_k only after each epoch
# err: an array that contains the value of the norm of the gradient at w_k for each iterate w_k only after each epoch
def sgd(ell, grad_ell, w0, data, batch_size, n_epochs):
    w = [w0]
    f_val = []
    grads = []
    err = []
    X, Y = data
    for epoch in range(n_epochs):
        for i in range(int(X.shape[0] / batch_size)):
            X_batch = X[i * batch_size:(i + 1) * batch_size, :]
            Y_batch = Y[i * batch_size:(i + 1) * batch_size]
            w.append(w[-1] - 1e-5 * grad_ell(w[-1], X_batch, Y_batch))
        f_val.append(ell(w[-1], X, Y))
        grads.append(grad_ell(w[-1], X, Y))
        err.append(np.linalg.norm(grad_ell(w[-1], X, Y), 2))
    return w, f_val, grads, err


# function that computes the gradient of the logistic loss
def grad_ell(w, X, Y):
    return np.dot(X.T, (1 / (1 + np.exp(-Y * np.dot(X, w))) - 1) * Y)


# function that computes the sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(float)))


# function that computes the logistic loss
def ell(w, X, Y):
    return np.sum(np.log(1 + np.exp(-Y * np.dot(X, w))))


def predict(w, X, threshold=0.5):
    return np.where(sigmoid(np.dot(X, w)) > threshold, 1, -1)


# load data.csv file which contains the MNIST dataset
data = pd.read_csv('data.csv', header=None)
# take in input two digits to select the data
X = data.iloc[:, 1:].values.T
Y = data.iloc[:, 0].values
print("Shape of X", X.shape)
print("Shape of Y", Y.shape)
digit1 = int(input('Enter the first digit: '))
digit2 = int(input('Enter the second digit: '))
digits = [digit1, digit2]
X = X[:, np.isin(Y, digits)]
Y = Y[np.isin(Y, digits)]
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")
# obtain the training set and the test set
N_train = int(X.shape[1] * 0.8)
N = X.shape[1]
idx = np.random.choice(N, N_train, replace=False)
X_train = X[:, idx].astype(np.float64)
Y_train = Y[idx].astype(np.float64)
X_test = np.delete(X, idx, axis=1)
Y_test = np.delete(Y, idx, axis=0)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of Y_test: {Y_test.shape}")
# test the logistic regression classifier with the stochastic gradient descent algorithm
# add a column of ones to the input data
Xhat_train = np.concatenate((np.ones((len(X_train.T), 1)), X_train.T), axis=1)
Xhat_test = np.concatenate((np.ones((len(X_test.T), 1)), X_test.T), axis=1)
# initialize the weight vector
w0 = np.random.rand(Xhat_train.shape[1])
# compute the stochastic gradient descent algorithm
w, f_val, grads, err = sgd(ell, grad_ell, w0, (Xhat_train, Y_train), 100, 100)
# compute the accuracy of the classifier
Y_pred = predict(w[-1], Xhat_test)
print(f"Accuracy: {np.sum(Y_pred == Y_test) / len(Y_test)}")


# compare the results with the results obtained with the gradient descent algorithm
# initialize the weight vector
w0 = np.random.rand(Xhat_train.shape[1])
# compute the gradient descent algorithm
def gd(ell, grad_ell, w0, data, kmax, tol):
    w = [w0]
    f_val = []
    grads = []
    err = []
    for k in range(kmax):
        w.append(w[-1] - 1e-5 * grad_ell(w[-1], *data))
        f_val.append(ell(w[-1], *data))
        grads.append(grad_ell(w[-1], *data))
        err.append(np.linalg.norm(grad_ell(w[-1], *data), 2))
        if err[-1] < tol:
            break
    return w, f_val, grads, err


w_gd, f_val_gd, grads_gd, err_gd = gd(ell, grad_ell, w0, (Xhat_train, Y_train), 1000, 1e-5)
Y_pred_gd = predict(w_gd[-1], Xhat_test)
print(f"Accuracy: {np.sum(Y_pred_gd == Y_test) / len(Y_test)}")

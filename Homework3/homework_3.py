import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def f(x):
    x_true = np.ones(10)
    A = np.vander(np.linspace(0, 1, 10), 10, increasing=True)
    b = A @ x_true
    c = 0.5
    return 0.5 * np.linalg.norm(A @ x - b) ** 2 + 0.5 * c * np.linalg.norm(x) ** 2


def grad_f(x):
    x_true = np.ones(10)
    A = np.vander(np.linspace(0, 1, 10), 10, increasing=True)
    b = A @ x_true
    c = 0.5
    return A.T @ (A @ x - b) + c * x


def backtracking(f, grad_f, x):
    alpha = 1
    c = 0.8
    tau = 0.25

    while f(x - alpha * grad_f(x)) > f(x) - c * alpha * np.linalg.norm(grad_f(x), 2) ** 2:
        alpha = tau * alpha

        if alpha < 1e-3:
            break
    return alpha


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


x0 = np.zeros(10)
x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, 1000, 1e-6, 1e-6, True)
plt.figure()
plt.plot(np.arange(k + 1), f_val)
plt.xlabel('k')
plt.ylabel('f(x_k)')
plt.title('f(x_k) for the fifth function')
plt.savefig('f(x_k)_fifth_function.png')
plt.clf()
x, k, f_val, grads, err = gradient_descent(f, grad_f, x0, 1000, 1e-6, 1e-6, False)
plt.figure()
plt.plot(np.arange(k + 1), f_val)
plt.xlabel('k')
plt.ylabel('f(x_k)')
plt.title('f(x_k) for the fifth function without backtracking')
plt.savefig('f(x_k)_fifth_function_without_backtracking.png')
plt.clf()


def l(w, D):
    return np.sum(np.log(1 + np.exp(-D * np.dot(D, w))))


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


def grad_ell(w, X, Y):
    return np.dot(X.T, (1 / (1 + np.exp(-Y * np.dot(X, w))) - 1) * Y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(float)))


def ell(w, X, Y):
    return np.sum(np.log(1 + np.exp(-Y * np.dot(X, w))))


def predict(w, X, threshold=0.5):
    return np.where(sigmoid(np.dot(X, w)) > threshold, 1, -1)


data = pd.read_csv('data.csv', header=None)
X = data.iloc[:, 1:].values.T
Y = data.iloc[:, 0].values
print("Shape of X", X.shape)
print("Shape of Y", Y.shape)
digit1 = int(input('Enter the first digit: '))
digit2 = int(input('Enter the second digit: '))
digits = [digit1, digit2]
X = X[:, np.isin(Y, digits)]
Y = Y[np.isin(Y, digits)]
N_train = int(X.shape[1] * 0.8)
N = X.shape[1]
idx = np.random.choice(N, N_train, replace=False)
X_train = X[:, idx].astype(np.float64)
Y_train = Y[idx].astype(np.float64)
X_test = np.delete(X, idx, axis=1)
Y_test = np.delete(Y, idx, axis=0)
Xhat_train = np.concatenate((np.ones((len(X_train.T), 1)), X_train.T), axis=1)
Xhat_test = np.concatenate((np.ones((len(X_test.T), 1)), X_test.T), axis=1)
w0 = np.random.rand(Xhat_train.shape[1])
w, f_val, grads, err = sgd(ell, grad_ell, w0, (Xhat_train, Y_train), 100, 100)
Y_pred = predict(w[-1], Xhat_test)
print(f"Accuracy: {np.sum(Y_pred == Y_test) / len(Y_test)}")
w0 = np.random.rand(Xhat_train.shape[1])
w_gd, f_val_gd, grads_gd, err_gd = gd(ell, grad_ell, w0, (Xhat_train, Y_train), 1000, 1e-5)
Y_pred_gd = predict(w_gd[-1], Xhat_test)
print(f"Accuracy: {np.sum(Y_pred_gd == Y_test) / len(Y_test)}")

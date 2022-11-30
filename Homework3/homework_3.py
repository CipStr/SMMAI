import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lambda_val = 0.5


def f(x, num):
    if num == 1:
        return (x[0] - 3) ** 2 + (x[1] - 1) ** 2
    elif num == 2:
        return 10 * (x[0] - 1) ** 2 + (x[1] - 2) ** 2
    elif num == 3:
        A = np.vander(np.linspace(0, 1, 50), 50, increasing=True)
        x_true = np.ones(50)
        b = A.dot(x_true)
        return 0.5 * np.linalg.norm(A.dot(x) - b) ** 2
    elif num == 4:
        A = np.vander(np.linspace(0, 1, 50), 50, increasing=True)
        x_true = np.ones(50)
        b = A.dot(x_true)
        return 0.5 * np.linalg.norm(A.dot(x) - b) ** 2 + lambda_val * np.linalg.norm(x) ** 2
    elif num == 5:
        return x[0] ** 4 + x[0] ** 3 - 2 * x[0] ** 2 - 2 * x[0]


def grad_f(x, num):
    if num == 1:
        return np.array([2 * (x[0] - 3), 2 * (x[1] - 1)])
    elif num == 2:
        return np.array([20 * (x[0] - 1), 2 * (x[1] - 2)])
    elif num == 3:
        A = np.vander(np.linspace(0, 1, 50), 50, increasing=True)
        x_true = np.ones(50)
        b = A.dot(x_true)
        return A.T.dot(A.dot(x) - b)
    elif num == 4:
        A = np.vander(np.linspace(0, 1, 50), 50, increasing=True)
        x_true = np.ones(50)
        b = A.dot(x_true)
        return A.T.dot(A.dot(x) - b) + lambda_val * x
    elif num == 5:
        return np.array([4 * x[0] ** 3 + 3 * x[0] ** 2 - 4 * x[0] - 2, 0])


def backtracking_line_search(f, grad_f, x, num):
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
    while f(x - alpha * grad_f(x, num), num) > f(x, num) - c * alpha * np.linalg.norm(grad_f(x, num), 2) ** 2:
        alpha = tau * alpha
        if alpha < 1e-3:
            break
    return alpha


def gradient_descent(f, grad_f, x0, tolf, tolx, kmax, alpha, xtrue, flag, num):
    f_val = np.zeros((kmax + 1,))
    grads = np.zeros((kmax + 1, len(grad_f(x0, num))))
    err = np.zeros((kmax + 1,))
    err_norm2 = np.zeros((kmax + 1,))
    x_tol = x0
    k = 0
    # compute iterations
    while k < kmax:
        if flag:
            alpha = backtracking_line_search(f, grad_f, x0, num)
        x = x0 - alpha * grad_f(x0, num)
        f_val[k] = f(x, num)
        grads[k, :] = grad_f(x, num)
        err[k] = np.linalg.norm(grads[k], 2)
        err_norm2[k] = np.linalg.norm(x - xtrue, 2)
        if err[k] < tolf * np.linalg.norm(grad_f(x_tol, num), 2) or err_norm2[k] < tolx:
            break
        else:
            x0 = x
        k += 1
    f_val = f_val[:k]
    grads = grads[:k, :]
    err = err[:k]
    err_norm2 = err_norm2[:k]
    return x, k, f_val, grads, err, err_norm2


def get_x0_xtrue(num):
    if num == 1:
        return np.array([0, 0]), np.array([3, 1])
    elif num == 2:
        return np.array([0, 0]), np.array([1, 2])
    elif num == 3:
        return np.zeros(50), np.ones(50)
    elif num == 4:
        return np.zeros(50), np.ones(50)
    elif num == 5:
        return (0,), 1


tolf = 0.005
tolx = 0.005
k_values = [[], [], [], [], []]
err_values = [[], [], [], [], []]
for i in range(1, 6):
    x0, xtrue = get_x0_xtrue(i)
    x_1, k_1, f_val, grads, err_1, err_norm2 = gradient_descent(f, grad_f, x0, tolf, tolx, 100, 0.1, xtrue, True, i)
    print(f"Number of iterations for the {i} function: {k_1} and the final value of x is {x_1}")
    x_2, k_2, f_val, grads, err_2, err_norm2 = gradient_descent(f, grad_f, x0, tolf, tolx, 100, 0.01, xtrue, False, i)
    print(f"Number of iterations for the {i} function: {k_2} and the final value of x is {x_2}")
    x_3, k_3, f_val, grads, err_3, err_norm2 = gradient_descent(f, grad_f, x0, tolf, tolx, 100, 0.05, xtrue, False, i)
    print(f"Number of iterations for the {i} function: {k_3} and the final value of x is {x_3}")
    k_array = [k_1, k_2, k_3]
    k_values[i - 1] = k_array
    err_array = [err_1, err_2, err_3]
    err_values[i - 1] = err_array
    plt.figure(figsize=(15, 5))
    for j in range(3):
        plt.subplot(1, 3, j + 1)
        plt.plot(np.linspace(1, k_array[j], k_array[j]), err_array[j])
        plt.xlabel("k")
        plt.ylabel("|| grad f(x_k) ||")
        plt.title(f"Function {i}")
    plt.savefig(f"function{i}.png")
    plt.clf()


tolf = 0.0005
tolx = 0.0005
x_values = []
k_values_2 = []
err_diff = []
for i in range(1, 6):
    x0, xtrue = get_x0_xtrue(i)
    x, k, f_val, grads, err, err_norm2 = gradient_descent(f, grad_f, x0, tolf, tolx, 100, 0.1, xtrue, True, i)
    x_values.append(x)
    k_values_2.append(k)
    err_diff.append(err_norm2)
    print(f"Value of || x_k - x_true || for each function: {err_norm2[k - 1]}")
print(f"Number of iterations for each function: {k_values_2}")
plt.figure(figsize=(16, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.plot(np.linspace(1, k_values_2[i], k_values_2[i]), err_diff[i])
    plt.xlabel("k")
    plt.ylabel("|| x_k - x_true ||")
    plt.title(f"Function {i + 1}")
plt.savefig("function1_5.png")
plt.clf()


x = np.linspace(-3, 3, 50)
result = np.zeros(len(x), )
x0, xtrue = get_x0_xtrue(5)
for i in range(len(x)):
    result[i] = gradient_descent(f, grad_f, np.array([x[i], ]), tolf, tolx, 100, 0.001, xtrue, False, 5)[0][0]
print(f"Convergence point for the function 5: {result}")
plt.plot(x, result, 'o')
plt.xlabel("Input interval")
plt.ylabel("Convergence point")
plt.title("Convergence point for the function 5 for different inputs")
plt.savefig("function5_with_inputs.png")
plt.clf()


def data_split(X, Y, train_size):
    d, N = X.shape
    # shuffle the data
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_train = idx[:train_size]
    idx_test = idx[train_size:]
    X_train = X[:, idx_train]
    Y_train = Y[idx_train]
    X_test = X[:, idx_test]
    Y_test = Y[idx_test]
    return X_train, Y_train, X_test, Y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grad_ell(w, X, Y):
    pred = param_f(w, X)
    return np.dot(X.T, (pred * (1 - pred)) * (pred - Y)) / X.shape[0]


def param_f(w, X):
    return sigmoid(np.dot(X, w))


def ell(w, X, Y):
    pred = param_f(w, X)
    return 1/len(Y) * (np.linalg.norm(np.subtract(pred, Y), 2) ** 2)


def predict(w, X, theshold = 0.5):
    pred = param_f(w, X)
    if pred >= theshold:
        return 1
    return 0


def stochastic_gradient_descent(ell, grad_ell, w0, data, batch_size, epochs):
    # intialize return values
    f_val = np.zeros((epochs,))
    grads = np.zeros((epochs,))
    err_values = np.zeros((epochs,))
    w = np.zeros((epochs,))
    alpha = 0.1
    for i in range(epochs):
        # get the batch
        batches = get_batch(data, batch_size)
        # compute the gradient
        for batch in batches:
            x, y = batch
            grad = grad_ell(w0, x, y)
            w = w0 - alpha * grad
            w0 = w
        f_val[i] = ell(w0, data[0], data[1])
    return w


def get_batch(data, batch_size):
    new_data = np.column_stack((data[0], data[1]))
    np.random.shuffle(new_data)
    batches = []
    for i in range(new_data.shape[0] // batch_size):
        batch = new_data[i * batch_size:(i + 1) * batch_size, :]
        X_batch = batch[:, :-1]
        Y_batch = batch[:, -1].reshape(-1, 1)
        batches.append((X_batch, Y_batch))
    return batches


data = pd.read_csv("data.csv")
print(f"Data shape: {data.shape}")
dataset = np.array(data)
X = dataset[:, 1:]
X = X.T
Y = dataset[:, 0]
digit1 = int(input("Enter the first digit: "))
digit2 = int(input("Enter the second digit: "))
i1 = (Y == digit1)
i2 = (Y == digit2)
X1 = X[:, i1]
X2 = X[:, i2]
Y1 = Y[i1]
Y2 = Y[i2]
X = np.concatenate((X1, X2), axis=1)
Y = np.concatenate((Y1, Y2), axis=0)
mean = np.mean(X, axis=0, keepdims=True)
std = np.std(X, axis=0, keepdims=True)
X = (X - mean) / std
Y[Y == digit1] = 0
Y[Y == digit2] = 1
Xhat = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
X_train, Y_train, X_test, Y_test = data_split(Xhat, Y, round(len(Y) * 0.8))
d, N = X_train.shape
Y_train = Y_train.reshape((N, 1))
data = (X_train.T, Y_train)
w0 = np.random.normal(0, 0.001, (d, 1))
w1 = stochastic_gradient_descent(ell, grad_ell, w0, data, 20, 10)
w2 = stochastic_gradient_descent(ell, grad_ell, w0, data, N, 10)
pred1 = np.round(param_f(w1, X_test.T))
pred2 = np.round(param_f(w2, X_test.T))


from sklearn.metrics import confusion_matrix, accuracy_score
print(f"Confusion matrix for batch size 20: {confusion_matrix(Y_test, pred1)}")
print(f"Confusion matrix for batch size N: {confusion_matrix(Y_test, pred2)}")


print(f"Accuracy for batch size 20: {accuracy_score(Y_test, pred1)}")
print(f"Accuracy for batch size N: {accuracy_score(Y_test, pred2)}")


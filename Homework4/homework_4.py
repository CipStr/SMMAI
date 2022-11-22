import numpy as np
import matplotlib.pyplot as plt


k_true = int(input("Please enter a positive integer k: "))
theta_true = np.ones(k_true).T


N = 10
a = int(input("Please enter a positive integer a: "))
b = int(input("Please enter a positive integer b (b>a): "))
X = np.random.uniform(a, b, N)


def vander(j, x):
    return x ** j


A = np.zeros((N, k_true))
for i in range(N):
    for j in range(k_true):
        A[i, j] = vander(j, X[i])
print("A vandermonde matrix = ", A)
sigma = float(input("Please enter a positive float sigma: "))
eps = np.random.normal(0, sigma, N)
Y = A @ theta_true + eps
print("Y = ", Y)
D = (X, Y)


def MLE(D, K):
    X, Y = D
    N = len(X)
    A = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            A[i, j] = vander(j, X[i])
    theta_MLE = np.linalg.inv(A.T @ A) @ A.T @ Y
    return theta_MLE


def average_error(theta, T):
    Xtest, Ytest = T
    Ntest = len(Xtest)
    K = len(theta)
    A = np.zeros((Ntest, K))
    for i in range(Ntest):
        for j in range(K):
            A[i, j] = vander(j, Xtest[i])
    error = (1 / Ntest) * (np.linalg.norm(A @ theta - Ytest) ** 2)
    return error


for k in range(1, 11):
    theta_MLE = MLE(D, k)
    print("MLE solution for k = ", k, " is ", theta_MLE)
    Xtest = np.linspace(a, b, 20)
    Ytest = np.zeros(20)
    for i in range(20):
        for j in range(k):
            Ytest[i] += theta_MLE[j] * vander(j, Xtest[i])
    plt.plot(X, Y, 'ro')
    # plot test points as green dots
    plt.plot(Xtest, Ytest, 'go')
    # plot the polynomial regressor f_theta(x) for the MLE solution
    plt.plot(Xtest, Ytest, 'b')
    plt.legend(['Training points', 'Test points', 'MLE solution'])
    plt.title('MLE solution for k = ' + str(k))
    plt.savefig('MLE_solution_k_' + str(k) + '.png')
    plt.clf()


for k in range(1, 6):
    theta_MLE = MLE(D, k)
    Xtest = np.linspace(a, b, 20)
    A_test = np.zeros((20, k))
    theta_true_test = np.ones(k).T
    eps_test = np.random.normal(0, sigma, 20)
    for i in range(20):
        for j in range(k):
            A_test[i, j] = vander(j, Xtest[i])
    Ytest = A_test @ theta_true_test + eps_test
    T = (Xtest, Ytest)
    training_error = average_error(theta_MLE, D)
    test_error = average_error(theta_MLE, T)
    plt.plot(k, training_error, 'ro')
    plt.plot(k, test_error, 'go')
    plt.legend(['Training error', 'Test error'])
    plt.title('Training error and test error as a function of k=' + str(k))
    plt.savefig('training_error_test_error_k=' + str(k) + '.png')
    plt.clf()


def MAP(D, K, l):
    X, Y = D
    N = len(X)
    A = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            A[i, j] = vander(j, X[i])
    theta_MAP = np.linalg.inv(A.T @ A + l * np.identity(K)) @ A.T @ Y
    return theta_MAP


for k in range(1, 11):
    theta_MAP = MAP(D, k, 0.1)
    print("MAP solution for k = ", k, " is ", theta_MAP)
    Xtest = np.linspace(a, b, 20)
    Ytest = np.zeros(20)
    for i in range(20):
        for j in range(k):
            Ytest[i] += theta_MAP[j] * vander(j, Xtest[i])
    plt.plot(X, Y, 'ro')
    plt.plot(Xtest, Ytest, 'go')
    plt.plot(Xtest, Ytest, 'b')
    plt.legend(['Training points', 'Test points', 'MAP solution'])
    plt.title('MAP solution for k = ' + str(k))
    plt.savefig('MAP solution for k = ' + str(k) + '.png')
    plt.clf()


k = 10
theta_MLE = MLE(D, k)
theta_MAP = []
for l in range(1, 11):
    theta_MAP.append(MAP(D, k, l))
Xtest = np.linspace(a, b, 20)
A_test = np.zeros((20, k))
theta_true_test = np.ones(k).T
eps_test = np.random.normal(0, sigma, 20)
for i in range(20):
    for j in range(k):
        A_test[i, j] = vander(j, Xtest[i])
Ytest = A_test @ theta_true_test + eps_test
T = (Xtest, Ytest)
MLE_test_error = average_error(theta_MLE, T)
MAP_test_error = []
for l in range(1, 11):
    MAP_test_error.append(average_error(theta_MAP[l - 1], T))
print("MLE test error = ", MLE_test_error)
print("MAP test error = ", MAP_test_error)
for l in range(1, 11):
    if MAP_test_error[l - 1] < MLE_test_error:
        print("MAP test error is lower than MLE test error for lambda = ", l)


MLE_err = []
MAP_err = []
for k in range(1, 11):
    theta_true_padded = np.zeros(len(theta_MLE))
    for i in range(k_true):
        theta_true_padded[i] = theta_true[i]
    MLE_err.append(np.linalg.norm(theta_MLE - theta_true_padded) / np.linalg.norm(theta_true_padded))
    for l in range(1, 4):
        MAP_err.append(np.linalg.norm(theta_MAP[l - 1] - theta_true_padded) / np.linalg.norm(theta_true_padded))
print("MLE err = ", MLE_err)
print("MAP err = ", MAP_err)


Y = np.zeros(N)
for i in range(N):
    lambda_poiss = 0
    for j in range(k_true):
        lambda_poiss += theta_true[j] * X[j]**j
    Y[i] = np.random.poisson(lambda_poiss)
D = (X, Y)
print("Y with Poisson distribution = ", Y)


theta_MLE = MLE(D, k_true)
print("MLE solution for Poisson distribution = ", theta_MLE)
# compute the MAP solution for the Poisson distribution
theta_MAP = MAP(D, k_true, 0.1)
print("MAP solution for Poisson distribution = ", theta_MAP)


MLE_training_error = []
MAP_training_error = []
for k in range(1, 11):
    theta_MLE = MLE(D, k)
    MLE_training_error.append(average_error(theta_MLE, D))
    theta_MAP = MAP(D, k, 0.1)
    MAP_training_error.append(average_error(theta_MAP, D))
print("MLE training error = ", MLE_training_error)
print("MAP training error = ", MAP_training_error)
for k in range(1, 11):
    if MAP_training_error[k - 1] < MLE_training_error[k - 1]:
        print("MAP training error is lower than MLE training error for k = ", k)







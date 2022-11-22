import numpy as np
import matplotlib.pyplot as plt

# Maximum likelihood estimation and Maximum a posteriori estimation

# let the user fix a positive integer k and define theta_true as a vector of ones of length k
k_true = int(input("Please enter a positive integer k: "))
theta_true = np.ones(k_true).T

# define an input dataset where there are N uniformely distributed random numbers between a and b, where a and b are
# user defined
N = 10
a = int(input("Please enter a positive integer a: "))
b = int(input("Please enter a positive integer b (b>a): "))
X = np.random.uniform(a, b, N)


# a function defining the classical Vandermonde matrix where vander(j, x) = x^(j-1)
def vander(j, x):
    return x ** j


# compute the matrix A of size N x k where A[i, j] = vander(j, X[i])
A = np.zeros((N, k_true))
for i in range(N):
    for j in range(k_true):
        A[i, j] = vander(j, X[i])
print("A vandermonde matrix = ", A)
# given a variance sigma^2, compute Y=A@theta_true + eps where eps is a standard normal gaussian noise N(0, sigma^2*I)
sigma = float(input("Please enter a positive float sigma: "))
eps = np.random.normal(0, sigma, N)
Y = A @ theta_true + eps
print("Y = ", Y)

# build a dataset D = (X, Y) such that theta_true is the best solution to the least squares problem
D = (X, Y)


# define a function that takes in input D and K and returns the MLE solution (with Gaussian assumption) to the least
# squares problem
# note that the loss function can be optimized using the normal equation
def MLE(D, K):
    X, Y = D
    N = len(X)
    A = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            A[i, j] = vander(j, X[i])
    theta_MLE = np.linalg.inv(A.T @ A) @ A.T @ Y
    return theta_MLE


# write a function that takes as input a set of K-dimensional parameter vector theta and a test set T = (Xtest, Ytest)
# and returns the average absolute error of the polynomial regressor f_theta(x) computed as:
# 1/Ntest * np.linalg.norm(f_theta(Xtest) - Ytest, ord=2)**2
# f_theta(x) = theta_1 + theta_2 * x + theta_3 * x^2 + ... + theta_K * x^(K-1)
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

# for different values of K, plot the training datapoints and the test datapoints with different colors,
# and visualize as a continuous line the polynomial regressor f_theta(x) for the MLE solution
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

# for increasing values of K, compute the training error and the test error, where the test set is
# generated as in the previous exercise
# plot the training error and the test error as a function of K
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
    # y axis is the error (starts from zero) and x axis is the value of k
    plt.plot(k, training_error, 'ro')
    plt.plot(k, test_error, 'go')
    plt.legend(['Training error', 'Test error'])
    plt.title('Training error and test error as a function of k=' + str(k))
    plt.savefig('training_error_test_error_k=' + str(k) + '.png')
    plt.clf()

# define a function that takes in input D, K and lambda>0 and returns the MAP solution (with Gaussian assumption) to
# the least squares problem note that the loss function can be optimized using the normal equation
def MAP(D, K, l):
    X, Y = D
    N = len(X)
    A = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            A[i, j] = vander(j, X[i])
    theta_MAP = np.linalg.inv(A.T @ A + l * np.identity(K)) @ A.T @ Y
    return theta_MAP

# for K lower, equal or greater than the correct test polynomial degree, plot the training datapoints and the test
# datapoints with different colors, and visualize as a continuous line the polynomial regressor f_theta(x) for the
# MAP solution
for k in range(1, 11):
    theta_MAP = MAP(D, k, 0.1)
    print("MAP solution for k = ", k, " is ", theta_MAP)
    Xtest = np.linspace(a, b, 20)
    Ytest = np.zeros(20)
    for i in range(20):
        for j in range(k):
            Ytest[i] += theta_MAP[j] * vander(j, Xtest[i])
    plt.plot(X, Y, 'ro')
    # plot test points as green dots
    plt.plot(Xtest, Ytest, 'go')
    # plot the polynomial regressor f_theta(x) for the MAP solution
    plt.plot(Xtest, Ytest, 'b')
    plt.legend(['Training points', 'Test points', 'MAP solution'])
    plt.title('MAP solution for k = ' + str(k))
    plt.savefig('MAP solution for k = ' + str(k) + '.png')
    plt.clf()

# for K being way greater than the correct test polynomial degree, compute the MLE and MAP solutions. Compare the
# test error of the two for different values of lambda in the MAP solution
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
# print the test error for the MLE and MAP solutions
print("MLE test error = ", MLE_test_error)
print("MAP test error = ", MAP_test_error)
# print for which value the MAP test error is lower than the MLE test error
for l in range(1, 11):
    if MAP_test_error[l - 1] < MLE_test_error:
        print("MAP test error is lower than MLE test error for lambda = ", l)

# for K greater than the correct test polynomial degree, define err(theta) as norm(theta-theta_true)/norm(theta_true),
# where theta_true has been padded with zeros to have the same dimension as theta. Compute the err(theta) for the
# MLE and MAP solutions for different values of lambda in the MAP solution
MLE_err = []
MAP_err = []
for k in range(1, 11):
    # pad theta_true with zeros to have the same dimension as theta_MLE but withouth replacing the first k elements
    theta_true_padded = np.zeros(len(theta_MLE))
    for i in range(k_true):
        theta_true_padded[i] = theta_true[i]
    MLE_err.append(np.linalg.norm(theta_MLE - theta_true_padded) / np.linalg.norm(theta_true_padded))
    for l in range(1, 4):
        MAP_err.append(np.linalg.norm(theta_MAP[l - 1] - theta_true_padded) / np.linalg.norm(theta_true_padded))
# print the err(theta) for the MLE and MAP solutions
print("MLE err = ", MLE_err)
print("MAP err = ", MAP_err)


# set yi as Poi(y| theta_true * xi + ... + theta_true * xi^(k-1) where Poi(y| lambda) is the Poisson distribution with
# parameter lambda.
# reuse k_true, N and X from the previous exercise
Y = np.zeros(N)
for i in range(N):
    lambda_poiss = 0
    for j in range(k_true):
        lambda_poiss += theta_true[j] * X[j]**j
    Y[i] = np.random.poisson(lambda_poiss)
D = (X, Y)
print("Y with Poisson distribution = ", Y)

# compute the MLE solution for the Poisson distribution
theta_MLE = MLE(D, k_true)
print("MLE solution for Poisson distribution = ", theta_MLE)
# compute the MAP solution for the Poisson distribution
theta_MAP = MAP(D, k_true, 0.1)
print("MAP solution for Poisson distribution = ", theta_MAP)

# repeat the same steps as in the previous exercise for the Poisson distribution
MLE_training_error = []
MAP_training_error = []
for k in range(1, 11):
    theta_MLE = MLE(D, k)
    MLE_training_error.append(average_error(theta_MLE, D))
    theta_MAP = MAP(D, k, 0.1)
    MAP_training_error.append(average_error(theta_MAP, D))
# print the training error for the MLE and MAP solutions
print("MLE training error = ", MLE_training_error)
print("MAP training error = ", MAP_training_error)
# print for which value the MAP training error is lower than the MLE training error
for k in range(1, 11):
    if MAP_training_error[k - 1] < MLE_training_error[k - 1]:
        print("MAP training error is lower than MLE training error for k = ", k)







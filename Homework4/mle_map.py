import numpy as np
import matplotlib.pyplot as plt


# Maximum likelihood estimation and Maximum a posteriori estimation

# function that computer the vandermonde matrix
def vander(X, k):
    N = len(X)
    A = np.zeros((N, k))
    for j in range(k):
        A[:, j] = X ** j
    return A


# let the user fix a positive integer k and define theta_true as a vector of ones of length k
# k_true = int(input("Please enter a positive integer k: "))
k_true = 5
theta_true = np.ones(k_true).T

# define an input dataset where there are N uniformely distributed random numbers between a and b, where a and b are
# user defined
N = 100
# a = int(input("Please enter a positive integer a: "))
# b = int(input("Please enter a positive integer b (b>a): "))
a = 0
b = 1
X = np.linspace(a, b, N)
A = vander(X, k_true)
# sigma = float(input("Please enter a positive float sigma: "))
sigma = 0.1
noise = np.random.normal(0, 1, N)
Y = A @ theta_true + sigma * noise
D = (X, Y)


# define a function that computes the MLE given a dataset D and a positive integer k
def MLE(D, K):
    X, Y = D
    A = vander(X, K)
    theta_mle = np.linalg.solve(A.T @ A, A.T @ Y)
    return theta_mle


# define a function that computes the MAP given a dataset D, a positive integer k, and a positive float lambda
def MAP(D, K, lam):
    X, Y = D
    A = vander(X, K)
    theta_map = np.linalg.solve(A.T @ A + lam * np.eye(K), A.T @ Y)
    return theta_map


# define a function that computes the error between the true theta and the MLE or MAP
def error(true, estimate):
    f_Xtest = vander(estimate[0], len(true)) @ true
    return (1 / len(estimate[0])) * np.linalg.norm(f_Xtest - estimate[1], 2) ** 2


# define a function that computes the polynomial regressor for MLE
def f_mle(D, K, theta):
    X, Y = D
    A = vander(X, K)
    return (1 / 2) * np.linalg.norm(A @ theta - Y, 2) ** 2


# define a function that computes the polynomial regressor for MAP
def f_map(D, K, theta, lmbd):
    X, Y = D
    A = vander(X, K)
    return (1 / 2) * np.linalg.norm(A @ theta - Y, 2) ** 2 + (lmbd / 2) * np.linalg.norm(theta, 2) ** 2


# define a function that computes the gradient (for MLE)
def grad_mle(D, K, theta):
    X, Y = D
    A = vander(X, K)
    return A.T @ (A @ theta - Y)


# define a function that computes the gradient (for MAP)
def grad_map(D, K, theta, lmbd):
    X, Y = D
    A = vander(X, K)
    return A.T @ (A @ theta - Y) + lmbd * theta


# define a function that computes the MLE but using stochastic gradient descent
def MLE_SGD(D, K, f, grad_f, batch_size, n_epochs):
    theta = np.zeros((K,))
    alpha = 0.001
    for i in range(n_epochs):
        mini_batches = create_mini_batches(D, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch
            theta -= alpha * grad_f(mini_batch, K, theta)
    return theta


# define a function that computes the MAP but using stochastic gradient descent
def MAP_SGD(D, K, f, grad_f, batch_size, n_epochs, lmbd):
    theta = np.zeros((K,))
    alpha = 0.001
    for i in range(n_epochs):
        mini_batches = create_mini_batches(D, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch
            theta -= alpha * grad_f(mini_batch, K, theta, lmbd)
    return theta


# a function that creates mini batches for SGD given a dataset D and a batch size
def create_mini_batches(D, minibatch_size):
    X, Y = D
    m = Y.shape[0]

    permutation = list(np.random.permutation(m - 1))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    minibatches = []
    number_of_minibatches = int(m / minibatch_size)

    for k in range(number_of_minibatches):
        minibatch_X = shuffled_X[k * minibatch_size:(k + 1) * minibatch_size]
        minibatch_Y = shuffled_Y[k * minibatch_size:(k + 1) * minibatch_size]
        minibatch_pair = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch_pair)

    return minibatches


# create the test dataset
X_test = np.linspace(a, b, 1000)
Y_test = vander(X_test, k_true) @ theta_true + sigma * np.random.normal(0, 1, 1000)

# compute the MLE and MAP for different values of k
plt.figure(figsize=(16, 16))
fig_idx = 1
k = [2, 5, 8, 12, 15, 20]
for i in k:
    theta = MLE((X, Y), i)
    f_Xtest = vander(X_test, i) @ theta
    plt.subplot(3, 2, fig_idx)
    plt.plot(X, Y, 'bo')
    plt.plot(X_test, Y_test, 'go')
    plt.plot(X_test, f_Xtest, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Training data', 'Test data', 'f_theta_MLE(x)'])
    plt.title('MLE Regression Model for K = ' + str(i))
    plt.grid()
    fig_idx += 1
plt.savefig('MLE.png')
plt.clf()

# plot the error for MLE for different values of k
err_test = np.zeros((16,))
err_train = np.zeros((16,))

for i in range(16):
    theta = MLE((X, Y), i + 3)
    # theta=MLE_SGD((X,Y),i+3,f_mle,grad_mle,X.size,100)
    err_test[i] = error(theta, (X_test, Y_test))
    err_train[i] = error(theta, (X, Y))

k = np.arange(3, 19, 1)
plt.plot(k, err_train)
plt.plot(k, err_test)
plt.legend(['Training data', 'Test data'])
plt.xlabel('K')
plt.ylabel('Error')
plt.title('Error of MLE Regression Model for increasing K')
plt.grid()
plt.savefig('MLE_error.png')
plt.clf()

# plot the MAP for different values of k and lambda
plt.figure(figsize=(16, 16))
k = [2, 4, 20]
lambda_array = [0.1, 5, 100]
idx = 1
for i in k:
    for l in lambda_array:
        theta = MAP((X, Y), i, l)
        f_Xtest = vander(X_test, i) @ theta
        plt.subplot(3, 3, idx)
        plt.plot(X, Y, 'bo')
        plt.plot(X_test, Y_test, 'go')
        plt.plot(X_test, f_Xtest, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Test data', 'f_theta_MAP(x)'])
        plt.title('MAP Regression Model for K = ' + str(i) + ' and lambda = ' + str(l))
        plt.grid()
        idx += 1
plt.savefig('MAP.png')
plt.clf()

# plot the MAP error for different values of k and constant lambda
plt.figure(figsize=(20, 20))
k = [2, 4, 7, 12, 20, 30, 50, 200]
lambda_value = 5  # lambda value for which we want to plot the error
idx = 1
for i in k:
    # when k>4 overfitting, when k<4 underfitting
    theta = MAP((X, Y), i, lambda_value)
    f_Xtest = vander(X_test, i) @ theta
    plt.subplot(4, 3, idx)
    plt.plot(X, Y, 'ro')
    plt.plot(X_test, Y_test, 'bo')
    plt.plot(X_test, f_Xtest, color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Training data', 'Test data', 'f_theta_MAP(x)'])
    plt.title('MAP Regression Model for K = ' + str(i) + ' and lambda = ' + str(lambda_value))
    plt.grid()
    idx += 1
plt.savefig('MAP_constant_lambda.png')
plt.clf()

# plot the MAP error for different values of lambda and constant k
plt.figure(figsize=(20, 20))
lambda_array = [0, 0.01, 0.1, 1, 5, 10, 30, 100]
k = 20
idx = 1
for i in lambda_array:
    # when k>4 overfitting, when k<4 underfitting
    theta = MAP((X, Y), k, i)
    f_Xtest = vander(X_test, k) @ theta
    plt.subplot(4, 3, idx)
    plt.plot(X, Y, 'ro')
    plt.plot(X_test, Y_test, 'bo')
    plt.plot(X_test, f_Xtest, color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Training data', 'Test data', 'f_theta_MAP(x)'])
    plt.title('MAP Regression Model for K = ' + str(k) + ' and lambda = ' + str(i))
    plt.grid()
    idx += 1
plt.savefig('MAP_constant_k.png')
plt.clf()

# print the errors for normal equations optimization
print('Error for normal equations optimization')
K = 20
lambda_array = [0.1, 0.5, 1, 5, 10, 50, 100, 200, 300]
theta_mle = MLE((X, Y), K)
print('MLE error for K = ', K, ': ', error(theta_mle, (X_test, Y_test)))
for i in lambda_array:
    theta_map = MAP((X, Y), K, i)
    print('MAP error for K = ', K, 'and lambda = ', i, ': ', error(theta_map, (X_test, Y_test)))

# print the errors for gradient descent optimization
print('Error for gradient descent optimization')
K = 20
lambda_array = [0.1, 0.5, 1, 5, 10, 50, 100, 200, 300]
# theta_mle=MLE_GD((X,Y),K,f_mle,grad_mle,0.00005,0.00005,100)
theta_mle = MLE_SGD((X, Y), K, f_mle, grad_mle, X.size, 100)
print('MLE error for K = ', K, ': ', error(theta_mle, (X_test, Y_test)))
for i in lambda_array:
    # theta_map = MAP_GD((X,Y),K,f_map,grad_map,0.00005,0.00005,100,i)
    theta_map = MAP_SGD((X, Y), K, f_map, grad_map, X.size, 100, i)
    print('MAP error for K = ', K, 'and lambda = ', i, ': ', error(theta_map, (X_test, Y_test)))

# print theta errors for normal equations optimization
print('Theta errors for normal equations optimization')
lambda_array = [0.1, 0.5, 1, 5, 10, 50, 100, 200, 300]
for m in range(5, 20):
    print('')
    theta_mle = MLE((X, Y), m)
    theta_true_zero = np.pad(theta_true, (0, m - len(theta_true)))
    print('MLE theta error (K=', m, ') : ',
          np.linalg.norm(theta_mle - theta_true_zero, 2) / np.linalg.norm(theta_true_zero, 2))
    for i in lambda_array:
        theta_map = MAP((X, Y), m, i)
        print('MAP theta error (K=', m, 'lambda=', i, ') : ',
              np.linalg.norm(theta_map - theta_true_zero, 2) / np.linalg.norm(theta_true_zero, 2))

# print theta errors for gradient descent optimization
print('Theta errors for gradient descent optimization')
for K in range(5, 20):
    print('')
    # theta_mle = MLE_GD((X,Y),K,f_mle,grad_mle,0.00005,0.00005,100)
    theta_mle = MLE_SGD((X, Y), K, f_mle, grad_mle, X.size, 100)
    theta_true_zero = np.pad(theta_true, (0, K - len(theta_true)))
    print('MLE theta error (K =', K, ') : ',
          np.linalg.norm(theta_mle - theta_true_zero, 2) / np.linalg.norm(theta_true_zero, 2))
    for i in lambda_array:
        # theta_map = MAP_GD((X,Y),K,f_map,grad_map,0.00005,0.00005,100,i)
        theta_map = MAP_SGD((X, Y), K, f_map, grad_map, X.size, 100, i)
        print('MAP theta error (K =', K, 'lambda =', i, ') : ',
              np.linalg.norm(theta_map - theta_true_zero, 2) / np.linalg.norm(theta_true_zero, 2))

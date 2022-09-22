import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def solution(A):  # A is a nxn matrix
    n = len(A)
    print(f"A = {A}")
    # create vector of ones x_true
    x_true = np.ones(n)
    print(f"x_true = {x_true}")
    # compute b
    b = A @ x_true
    print(f"b = {b}")
    # compute condition number in 2-norm of A
    cond_A = np.linalg.cond(A, 2)
    print(f"cond_A = {cond_A}")
    if cond_A > 1e10:
        print("A is ill-conditioned")
    else:
        print("A is well-conditioned")
    # compute condition number in inf-norm of A
    cond_A_inf = np.linalg.cond(A, np.inf)
    print(f"cond_A_inf = {cond_A_inf}")
    if cond_A_inf > 1e10:
        print("A is ill-conditioned")
    else:
        print("A is well-conditioned")

    # compute x
    x = np.linalg.solve(A, b)
    print(f"x = {x}")
    # compute relative error
    rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    print(f"rel_err = {rel_err}")
    return rel_err, cond_A, cond_A_inf


# create results map
results_rand = {}
# create a random matrix A with size varying from 10 to 100 with step 10
for n in range(10, 110, 10):
    A = np.random.rand(n, n)
    rel_err, cond_A, cond_A_inf = solution(A)
    # save the results in a map with key n
    results_rand[n] = (rel_err, cond_A, cond_A_inf)

# plot the results, one plot for relative error, one for condition number in 2-norm and one for condition number in
# inf-norm
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(list(results_rand.keys()), [results_rand[key][0] for key in results_rand.keys()])
ax[0].set_title("Relative error")
ax[1].plot(list(results_rand.keys()), [results_rand[key][1] for key in results_rand.keys()])
ax[1].set_title("Condition number in 2-norm")
ax[2].plot(list(results_rand.keys()), [results_rand[key][2] for key in results_rand.keys()])
ax[2].set_title("Condition number in inf-norm")
plt.savefig("linear_systems_rand.png")

# create results map
results_vandermonde = {}
# create a vandermonde matrix A with size varying from 5 to 30 with step 5
for n in range(5, 35, 5):
    A = np.vander(np.linspace(1, n, n))
    rel_err, cond_A, cond_A_inf = solution(A)
    # save the results in a map with key n
    results_vandermonde[n] = (rel_err, cond_A, cond_A_inf)

# plot the results, one plot for relative error, one for condition number in 2-norm and one for condition number in
# inf-norm
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(list(results_vandermonde.keys()), [results_vandermonde[key][0] for key in results_vandermonde.keys()])
ax[0].set_title("Relative error")
ax[1].plot(list(results_vandermonde.keys()), [results_vandermonde[key][1] for key in results_vandermonde.keys()])
ax[1].set_title("Condition number in 2-norm")
ax[2].plot(list(results_vandermonde.keys()), [results_vandermonde[key][2] for key in results_vandermonde.keys()])
ax[2].set_title("Condition number in inf-norm")
plt.savefig("linear_systems_vandermonde.png")

# create results map
results_hilbert = {}
# create a hilbert matrix A with size varying from 4 to 12 with step 1
for n in range(4, 13, 1):
    A = sp.linalg.hilbert(n)
    rel_err, cond_A, cond_A_inf = solution(A)
    # save the results in a map with key n
    results_hilbert[n] = (rel_err, cond_A, cond_A_inf)

# plot the results, one plot for relative error, one for condition number in 2-norm and one for condition number in
# inf-norm
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(list(results_hilbert.keys()), [results_hilbert[key][0] for key in results_hilbert.keys()])
ax[0].set_title("Relative error")
ax[1].plot(list(results_hilbert.keys()), [results_hilbert[key][1] for key in results_hilbert.keys()])
ax[1].set_title("Condition number in 2-norm")
ax[2].plot(list(results_hilbert.keys()), [results_hilbert[key][2] for key in results_hilbert.keys()])
ax[2].set_title("Condition number in inf-norm")
plt.savefig("linear_systems_hilbert.png")


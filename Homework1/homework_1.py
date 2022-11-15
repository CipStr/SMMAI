import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def sol(A, x_true):
    b = A @ x_true
    cond_A_2 = np.linalg.cond(A, 2)
    cond_A_inf = np.linalg.cond(A, np.inf)
    ill_cond_2 = cond_A_2 > 1e10
    ill_cond_inf = cond_A_inf > 1e10
    if ill_cond_2:
        print(f"The condition number in 2-norm of A is {cond_A_2} and A is ill-conditioned")
    if ill_cond_inf:
        print(f"The condition number in inf-norm of A is {cond_A_inf} and A is ill-conditioned")

    x = np.linalg.solve(A, b)
    rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    return rel_err, cond_A_2, cond_A_inf


results_rand = {}
for n in range(10, 110, 10):
    A = np.random.rand(n, n)
    x_true = np.ones(n)
    rel_err, cond_A_2, cond_A_inf = sol(A, x_true.T)
    results_rand[n] = (rel_err, cond_A_2, cond_A_inf)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(list(results_rand.keys()), [results_rand[key][0] for key in results_rand.keys()])
ax[0].set_title("Relative error")
ax[1].plot(list(results_rand.keys()), [results_rand[key][1] for key in results_rand.keys()])
ax[1].set_title("Condition number in 2-norm")
ax[2].plot(list(results_rand.keys()), [results_rand[key][2] for key in results_rand.keys()])
ax[2].set_title("Condition number in inf-norm")
plt.savefig("linear_systems_rand.png")

results_vandermonde = {}
for n in range(5,35,5):
    A = np.vander(np.linspace(1, n, n), increasing=True)
    x_true = np.arange(1, n+1)
    rel_err, cond_A_2, cond_A_inf = sol(A, x_true.T)
    results_vandermonde[n] = (rel_err, cond_A_2, cond_A_inf)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(list(results_vandermonde.keys()), [results_vandermonde[key][0] for key in results_vandermonde.keys()])
ax[0].set_title("Relative error")
ax[1].plot(list(results_vandermonde.keys()), [results_vandermonde[key][1] for key in results_vandermonde.keys()])
ax[1].set_title("Condition number in 2-norm")
ax[2].plot(list(results_vandermonde.keys()), [results_vandermonde[key][2] for key in results_vandermonde.keys()])
ax[2].set_title("Condition number in inf-norm")
plt.savefig("linear_systems_vandermonde.png")

results_hilbert = {}
for n in range(4, 13):
    A = sp.linalg.hilbert(n)
    x_true = np.ones(n)
    rel_err, cond_A_2, cond_A_inf = sol(A, x_true.T)
    results_hilbert[n] = (rel_err, cond_A_2, cond_A_inf)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(list(results_hilbert.keys()), [results_hilbert[key][0] for key in results_hilbert.keys()])
ax[0].set_title("Relative error")
ax[1].plot(list(results_hilbert.keys()), [results_hilbert[key][1] for key in results_hilbert.keys()])
ax[1].set_title("Condition number in 2-norm")
ax[2].plot(list(results_hilbert.keys()), [results_hilbert[key][2] for key in results_hilbert.keys()])
ax[2].set_title("Condition number in inf-norm")
plt.savefig("linear_systems_hilbert.png")

eps = 1
while 1 + eps > 1:
    eps /= 2
eps *= 2
print(f"Machine epsilon is {eps}")

for n in range(1, 1000001):
    a = (1 + 1/n)**n
    print(f"n = {n}, a = {a}, e = {np.e}, error = {np.abs(a - np.e)}")

A = np.array([[4,2], [1,3]])
B = np.array([[4,2], [2,1]])
print(f"The rank of A is {np.linalg.matrix_rank(A)}")
print(f"The rank of B is {np.linalg.matrix_rank(B)}")
if np.linalg.matrix_rank(A) == A.shape[0]:
    print("A is full rank")
if np.linalg.matrix_rank(B) == B.shape[0]:
    print("B is full rank")
print(f"The eigenvalues of A are {np.linalg.eigvals(A)}")
print(f"The eigenvalues of B are {np.linalg.eigvals(B)}")
for n in range(1, 5):
    C = np.random.randint(0, 10, (n, n))
    print(f"C = {C}")
    rank_C = np.linalg.matrix_rank(C)
    print(f"rank_C = {rank_C}")
    eig_C = np.linalg.eigvals(C)
    print(f"eig_C = {eig_C}")
    print(f"C is full rank: {rank_C == C.shape[0]}")
    C[n-1, :] = C[0, :]
    print(f"C = {C}")
    rank_C = np.linalg.matrix_rank(C)
    print(f"rank_C = {rank_C}")
    eig_C = np.linalg.eigvals(C)
    print(f"eig_C = {eig_C}")
    print(f"C is full rank: {rank_C == C.shape[0]}")



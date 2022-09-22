import numpy as np
import math as m

# compute epsilon, the smallest number such that 1 + epsilon != 1
epsilon = 1
while 1 + epsilon != 1:
    epsilon /= 2
print(f"epsilon = {epsilon}")

# compute the euler constant e on the sequence a_n = (1 + 1/n)^n
e = 1
# get the real value of the euler constant
e_real = m.e
print(f"e_real = {e_real}")

# compute a_n = (1 + 1/n)^n for n in range 1 to 1000
for n in range(100000001, 1000000001):
    a_n = (1 + 1/n)**n
    # compute the relative error
    rel_err = abs(a_n - e_real) / abs(e_real)
    # if the relative error is smaller than epsilon, print the value of n
    if rel_err < epsilon:
        print(f"n = {n}")
        break
    # update e
    e = a_n

print(f"e = {e}")
print("The euler constant is computed with the sequence a_n = (1 + 1/n)^n,\n"
      "where n is the number of terms in the sequence. What happens if n is a very large number?\n",
      "The relative error is smaller than epsilon, but the value of e is not the real value of the euler constant.\n")

# A matrix 2x2 with entries 4,2,1,3
A = np.array([[4, 2], [1, 3]])
# B matrix 2x2 with entries 4,2,2,1
B = np.array([[4, 2], [2, 1]])

# compute the rank of A
rank_A = np.linalg.matrix_rank(A)
print(f"rank_A = {rank_A}")
# compute the rank of B
rank_B = np.linalg.matrix_rank(B)
print(f"rank_B = {rank_B}")

# compute the eigenvalues of A
eig_A = np.linalg.eigvals(A)
print(f"eig_A = {eig_A}")
# compute the eigenvalues of B
eig_B = np.linalg.eigvals(B)
print(f"eig_B = {eig_B}")

# are A and B full rank?
print(f"A is full rank: {rank_A == 2}")
print(f"B is full rank: {rank_B == 2}")

# matrix C nxn with random integers in range 0 to 10
for n in range(1, 5):
    C = np.random.randint(0, 10, (n, n))
    print(f"C = {C}")
    # compute the rank of C
    rank_C = np.linalg.matrix_rank(C)
    print(f"rank_C = {rank_C}")
    # compute the eigenvalues of C
    eig_C = np.linalg.eigvals(C)
    print(f"eig_C = {eig_C}")
    # is C full rank?
    print(f"C is full rank: {rank_C == n}")
    # make C have rank n-1 by setting the last row equal to the first row
    C[n-1, :] = C[0, :]
    print(f"C = {C}")
    # compute the rank of C
    rank_C = np.linalg.matrix_rank(C)
    print(f"rank_C = {rank_C}")
    # compute the eigenvalues of C
    eig_C = np.linalg.eigvals(C)
    print(f"eig_C = {eig_C}")
    # is C full rank?
    print(f"C is full rank: {rank_C == n}")

print("The rank of a matrix is the number of linearly independent rows or columns.\n"
      "The rank of a matrix is equal to the number of eigenvalues of the matrix that are not zero.\n"
      "If the rank of a matrix is less than the number of rows or columns, the matrix is not full rank.\n")





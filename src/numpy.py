try: import legate.numpy as np
except: import numpy as np

n = 1000
# Generate a random n x n linear system
A = np.random.rand(n, n)
b = np.random.rand(n)

# initalize solution
x = np.zeros(b.shape)
# extrat diagnal elements
d = np.diag(A)
# non-diagnal elements
R = A-np.diag(d)

# Jacobi iteration x_{i+1} \gets (b-R x_{i}) D^{-1}
for _ in range(n):
    x = (b - np.dot(R, x)) / d

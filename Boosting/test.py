import numpy as np

y = np.array([1,0,1,1])
z = np.ones_like(y)
print(z)
z[y<1] = -1
print(z)
print(z*z)
import numpy as np
import matplotlib.pyplot as plt

D = np.random.random((2, 100))
plt.scatter(D[0], D[1])
plt.xlim(0,0.5)
plt.show()
# Jonathan You
# ITP 449 Summer 2022
# HW 4
# Question 1

import matplotlib.pyplot as plt
import numpy as np
import random

x = np.array([])
y = np.array([])

for counter in range(1, 201,1):
    x = np.append(x, random.randrange(1, 200, 1))
    y = np.append(y, random.randrange(1, 200, 1))

plt.scatter(x,y, color='red', s=20)
plt.xlabel('Random integer', color='b')
plt.ylabel('Random integer', color='b')
plt.title('Scatter of random integers', color = 'g')

plt.show()
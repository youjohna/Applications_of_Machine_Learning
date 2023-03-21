# Jonathan You
# ITP 449 Summer 2022
# HW 4
# Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

myDF = pd.read_csv('temp_data.csv')
print(myDF)
plt.plot(myDF['Year'], myDF['Value'], color = 'r', marker ='o', linestyle='--', markersize=5)
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly')
plt.title('Global Temperature')
plt.show()
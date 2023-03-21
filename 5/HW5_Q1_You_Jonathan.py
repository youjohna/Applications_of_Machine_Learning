# Jonathan You
# ITP 449 Summer 2022
# HW5
# Question 1
# Reading the mtcars.csv and setting the index to be car names

import pandas as pd
import numpy as np

frame = pd.read_csv('mtcars.csv')
print(frame)

frame.set_index(frame['Car Name'], inplace=True)
print(frame)
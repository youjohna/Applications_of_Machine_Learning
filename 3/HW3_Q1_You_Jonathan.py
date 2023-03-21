# Jonathan You
# ITP 449 Summer 2022
# HW 4
# Question 1

import numpy as np
import pandas as pd

# Part 1
myDict = {'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
          'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
          'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes'],
          'score': [12.5, 9, 16.5, np.nan, 9.0, 20.0, 14.5, np.nan, 8.0, 19.0]
}

myPD = pd.DataFrame(myDict)
print(myPD)

# Part 2
print()
print(myPD[['name', 'attempts']])

# Part 3
print()
print(myPD.loc[(np.logical_and((myPD['attempts'] == 1), (myPD['qualify'] == 'yes'))), ['name', 'score']])
# myPD.loc searches for True rows, and then print the column values for those rows

# Part 4
print()
myPD = myPD.fillna(0)
print(myPD)

# Part 5
print()
print(myPD.sort_values(by=['attempts', 'score'], ascending=[True, False], inplace=False))


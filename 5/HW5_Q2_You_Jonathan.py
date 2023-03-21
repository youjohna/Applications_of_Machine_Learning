# Jonathan You
# ITP 449 Summer 2022
# HW5
# Question 2
# Created a Dataframe and printed data in specified/desired formats

import pandas as pd
import numpy as np

frame = pd.read_csv('mtcars.csv')
frame.set_index(frame['Car Name'], inplace=True)

# Part 1
myDF = frame[['cyl', 'gear','hp','mpg']].copy()
myDF.rename(columns={'cyl': 'Cylinders',
                     'gear': 'Gear',
                     'hp': 'Horsepower',
                     'mpg': 'Miles per Gallon'}, inplace=True)
print(myDF)
print()

# Part 2
print()
myDF['Powerful'] = myDF['Horsepower']>=110
print(myDF)

# Part 3
print()
del myDF['Horsepower']
print(myDF)

# Part 4
print()
myDF['Horsepower'] = frame['hp'].copy()
myDF = myDF[['Cylinders', 'Gear', 'Horsepower', 'Miles per Gallon', 'Powerful']]
print(myDF[myDF['Miles per Gallon'] > 25.0].sort_values(by='Horsepower', ascending=False))

# Part 5
print()
myDF2 = myDF[myDF['Powerful'] == True]
print(myDF2.loc[myDF2['Miles per Gallon'].idxmax(), ['Cylinders', 'Gear', 'Horsepower', 'Miles per Gallon', 'Powerful']])



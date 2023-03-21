# Jonathan You
# ITP 449 Summer 2022
# HW 4
# Question 2

import numpy as np
import pandas as pd
import csv
from statistics import median
from statistics import mean

# Part 1
with open('Trojans_roster.csv') as csv_file:
    myDF = pd.read_csv('Trojans_roster.csv')
print(myDF)

# Part 2
print()
myDF.set_index(myDF['#'], inplace=True)
print(myDF)

# Part 3
print()
del myDF['LAST SCHOOL']
print(myDF)

# Part 4
print()
print(myDF.loc[myDF['POS.'] == 'QB', 'NAME'])

# Part 5
print()
print(myDF.loc[myDF['HT.'].idxmax(), ['NAME','POS.','HT.','WT.']])

# Part 6
print()
print(len(myDF['HOMETOWN']=='LOS ANGELES, CA'))

# Part 7 ---------------NOTE FINISHED-----------------
print()
print(myDF.sort_values('WT.').tail(3))

# Part 8
print()
myDF['BMI'] = np.array(703*myDF['WT.']/myDF['HT.']**2)
print(myDF)

# Part 9
print()
print("Weight Mean:"+str(mean(myDF['WT.']))+"\nWeight Median: "+str(median(myDF['WT.']))+"\nHeight Mean: "+str(mean(myDF['HT.']))+"\nHeight Median: "+str(median(myDF['HT.']))+"\nBMI Mean: "+str(mean(myDF['BMI']))+"\nBMI Median: "+str(median(myDF['BMI'])))

# Part 10
print()
print(myDF.groupby('POS.')[['HT.', 'WT.', 'BMI']].mean())
print()
print(myDF.groupby('POS.')[['HT.', 'WT.', 'BMI']].median())

# Part 11
print()
print(myDF.groupby('POS.')['NAME'].count())

# Part 12
print()
print(myDF.loc[myDF['BMI'] < mean(myDF['BMI']), ['NAME']])

# Part 13
print()
print(myDF['#'].unique())


# Jonathan You
# ITP 449 Summer 2022
# HW 4
# Question 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loan = float(input('Loan Amount: '))
annual_interest_rate = float(input('Annual Interest rate: '))
years = int(input('Years: '))

months = years*12
interestm = annual_interest_rate/12
value = (1+interestm)**months

payment = (loan*interestm*value)/(value-1)
print("Your monthly payment is $"+str(payment))

x1 = np.arange(0, months+1, 1)
y1 = (-100/months)*x1+100

x2 = np.arange(0, months+1, 1)
y2 = (-loan/months)*x2+loan

myFig = plt.figure()
myAx1 = myFig.add_subplot(1, 2, 1)
myAx2 = myFig.add_subplot(1, 2, 2)

myAx1.plot(x1,y1, markersize=2, marker='o')
myAx1.set_xlabel('Month')
myAx1.set_ylabel('Interest Paid')
myAx2.plot(x2,y2, markersize=2, marker='o')
myAx2.set_xlabel('Month')
myAx2.set_ylabel('Loan Balance')
plt.show()

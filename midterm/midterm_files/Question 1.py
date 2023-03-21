import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
loan = int(input('Loan amount:\n'))
interesty = float(input('Interest Rate:\n'))
years = int(input('Years:\n'))
extra = int(input('Enter the amount of your extra payment each month:\n'))

months = years*12
interestm = interesty/12/100
value = (1+interestm)**months

payment = (loan*interestm*(1+interestm)**months)/((1+interestm)**months-1)
print("Your monthly payment is $"+str(payment))

balance = loan
MonthArray = []
InterestArray = []
BalanceArray = []
ExtraArray = []
balanceCheck = True

while balanceCheck == True:
    for month in range(1, months+1, 1):
        interest = balance * interestm
        balance = balance + interest - payment - extra
        MonthArray.append(month)
        InterestArray.append(round(interest, 2))
        ExtraArray.append(extra)
        if balance <= 0:
            balanceCheck = False
            balance = 0
            BalanceArray.append(balance)
            break
        else:
            BalanceArray.append(round(balance, 2))

myDF = pd.DataFrame()
myDF['Month'] = MonthArray
myDF['Interest'] = InterestArray
myDF['Balance'] = BalanceArray
myDF['Extra'] = ExtraArray
myDF.set_index('Month')


print(myDF)
print('You will pay off your loans in ', str(months), ' months')

fig = plt.figure(figsize=(11,5))

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.plot(myDF['Month'], myDF['Interest'], marker='.', color='b')
ax1.set_xlabel('Month')
ax1.set_ylabel('Interest Paid')


ax2.plot(myDF['Month'], myDF['Balance'], marker='.', color='r')
ax2.set_xlabel('Month')
ax2.set_ylabel('Loan Balance')


plt.show()

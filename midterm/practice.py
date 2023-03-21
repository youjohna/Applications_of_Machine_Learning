import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

'''
# Question 1
print('Multiplication Table\n\n')
num = input("Please enter a whole number")
symbol = 'X'
for index in range(0,13):
     product = index*int(num)
     msg = ""
     if index<10:
          msg = " "
          msg = msg+str(index)
     else:
          msg = str(index)
     print(msg+symbol+num+"="+str(product))
print("Math is fun!")
'''

'''
# Question 2
loan = int(input('Loan amount:\n'))
interesty = int(input('Interest Rate:\n'))
years = int(input('Years:\n'))

months = years*12
interestm = interesty/12
value = (1+interestm)**months

payment = (loan*interestm*(1+interestm)**months)/((1+interestm)**months-1)
print("Your monthly payment is $"+str(payment))

loanPD = pd.DataFrame()
#loanPD['Month'] = np.arange(1, months+1, 1)
#loanPD['Interest'] = np.arrange(100,)
loanPD['Balance'] = np.arange(loan-payment, 0, -payment)
#loanPD.set_index('Month')
print(loanPD)
'''

# Question 3
org = pd.read_csv('mtcars(1).csv')
org.set_index(org['Car Name'], inplace=True)
myDF = org[['cyl', 'gear', 'hp', 'mpg']].copy()
myDF.rename(columns={
            'cyl': 'Cylinder',
            'gear': 'Gear',
            'hp': 'Horsepower',
            'mpg': 'Miles per Gallon'}, inplace=True)

print(myDF.sort_values(by=['Horsepower', 'Miles per Gallon'], ascending=[False, True]))
print()
print(myDF[myDF['Gear'] == 4])
print()
print(myDF[myDF['Miles per Gallon'] > 20])


figure1 = plt.figure(1)
myAx1 = plt.subplot(2, 2, 1)
myAx2 = plt.subplot(2, 2, 2)
myAx3 = plt.subplot(2, 2, 3)
myAx4 = plt.subplot(2, 2, 4)

myAx1.scatter(myDF['Horsepower'], myDF['Miles per Gallon'], s=7)
# plt.xticks(rotation='vertical')
myAx2.hist(myDF['Cylinder'], bins=20)
myAx3.bar(myDF.index, myDF['Gear'])

# sb.catplot(data=myDF, y='Horsepower', figure=figure1)

plt.show()



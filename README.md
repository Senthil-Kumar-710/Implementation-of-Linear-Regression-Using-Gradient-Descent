# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import necessary library.
2. Create new variable to read a CSV file.
3. Using training and test values on data-set,predict the linear line.
4. Print the program.

## Program:

'''
Program to implement the linear regression using gradient descent.
Developed by: Senthil Kumar S
RegisterNumber:  212221230091
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('student_scores - student_scores.csv')
dataset.head()
dataset.tail()
x = dataset.iloc[:,:-1].values    #.iloc[:,start_col,end_col]
y = dataset.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# For training data:
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# For testing data:
plt.scatter(x_test,y_test,color='orange')
plt.plot(x_test,regressor.predict(x_test),color='green')
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## Output:

## Training Data:

![train](https://user-images.githubusercontent.com/93860256/162005753-ad3e76c3-22f8-4bcc-b550-74339045b08e.PNG)


## Testing Data:

![test](https://user-images.githubusercontent.com/93860256/162005820-77d0a7cb-5150-44d6-affa-64ae680440e4.PNG)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

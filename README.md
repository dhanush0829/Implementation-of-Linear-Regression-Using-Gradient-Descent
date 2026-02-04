# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```py
/*
Program to implement the linear regression using gradient descent.
Developed by: Dhanush M
RegisterNumber: 212225230051
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\acer\Downloads\DATASET-20260131\50_Startups.csv")
x_data = data.iloc[:, 0].values   
y_data = data.iloc[:, 4].values   

plt.scatter(x_data, y_data)
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

def computeCost(x, y, theta):
    m = len(y)
    h = x.dot(theta)
    return (1/(2*m)) * np.sum((h - y)**2)

data_n = data.values
m = len(y_data)
x = np.c_[np.ones(m), x_data]
y = y_data.reshape(m, 1)
theta = np.zeros((2,1))

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = x.dot(theta)
        error = x.T.dot(predictions - y)
        theta -= (alpha / m) * error
        J_history.append(computeCost(x, y, theta))

    return theta, J_history

theta, J_history = gradientDescent(x, y, theta, 0.01, 1500)

print("h(x) = {} + {}x".format(round(theta[0,0],2), round(theta[1,0],2)))

plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function")
plt.figure()
plt.scatter(x_data, y_data)
x_range = np.linspace(min(x_data), max(x_data), 100)
y_pred = theta[0][0] + theta[1][0] * x_range
plt.plot(x_range, y_pred, color='r')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")
plt.show()
```

## Output:
<img width="808" height="576" alt="image" src="https://github.com/user-attachments/assets/42a80ca5-36b5-479c-8bca-dad4b002c054" />
<img width="784" height="513" alt="image" src="https://github.com/user-attachments/assets/881418d6-f6e2-4782-af72-ffa9c9181ff3" />
<img width="749" height="516" alt="image" src="https://github.com/user-attachments/assets/92f3eb80-6790-48de-8c01-762c7eab0f41" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

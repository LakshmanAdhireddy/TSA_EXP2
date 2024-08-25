# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
## Date:
## Developed By:Lakshman
## Reg no:212222240001
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1.Import necessary libraries (NumPy, Matplotlib)

2.Load the dataset

3.Calculate the linear trend values using least square method

4.Calculate the polynomial trend values using least square method

5.End the program
### PROGRAM:
## A - LINEAR TREND ESTIMATION
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv('infy_stock-Copy1.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data['Date'] = data['Date'].apply(lambda x: x.toordinal())
X = data['Date'].values.reshape(-1, 1)
y = data['Volume'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
data['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Volume'],label='Original Data')
plt.plot(data['Date'], data['Linear_Trend'], color='orange', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
~~~

## B- POLYNOMIAL TREND ESTIMATION
~~~
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
data[' Volume'] = poly_model.predict(X_poly)
plt.figure(figsize=(10,6))
plt.bar(data['Date'], data['Volume'], label='Original Data', alpha=0.6)
plt.plot(data['Date'], data[' Volume'],color='yellow', label='Poly Trend(Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
~~~

### OUTPUT
## A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/30dfa6c3-33da-40ec-854b-deaad0f2804a)

## B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/7262de3b-1fce-4c85-9e0c-20ee22631ec1)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.

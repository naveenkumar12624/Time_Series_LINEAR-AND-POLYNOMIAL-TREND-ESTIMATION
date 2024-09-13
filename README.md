### Developed by: Naveen Kumar S
### Register Number: 212221240033
### Date:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:

1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program

### PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the NVIDIA dataset
data = pd.read_csv('C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv')

# Convert 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Prepare data for modeling
data = data.resample('Y').mean()  # Resample to yearly frequency and take mean of 'Close' prices
X = np.arange(len(data)).reshape(-1, 1)  # Feature for linear regression
y = data['Close'].values  # Target variable
```
A - LINEAR TREND ESTIMATION
```python
# Linear Trend Estimation
regressor = LinearRegression()
regressor.fit(X, y)
y_linear_predict = regressor.predict(X)

# Plotting Linear Trend Estimation
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(data.index, y, label='Actual Data', color='black')
plt.plot(data.index, y_linear_predict, label='Linear Trend', color='blue')
plt.title('A - Linear Trend Estimation')
plt.xlabel('Year')
plt.ylabel('Average Close Price')
plt.grid(True)
plt.legend()

```

B - POLYNOMIAL TREND ESTIMATION:
```python
# Polynomial Trend Estimation
plt.figure(figsize=(16, 5))

# Polynomial trend for degree 2
degree = 2
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X)
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly, y)
y_predict_poly_2 = regressor_poly.predict(X_poly)

# Polynomial trend for degree 3
degree_3 = 3
poly_reg_3 = PolynomialFeatures(degree=degree_3)
X_poly_3 = poly_reg_3.fit_transform(X)
regressor_poly_3 = LinearRegression()
regressor_poly_3.fit(X_poly_3, y)
y_predict_poly_3 = regressor_poly_3.predict(X_poly_3)

# Plotting Polynomial Trend Estimation
plt.subplot(1, 2, 2)
plt.plot(data.index, y, label='Actual Data', color='black')
plt.plot(data.index, y_predict_poly_2, label=f'Polynomial Trend (Degree {degree})', linestyle='-.', color='green')
plt.plot(data.index, y_predict_poly_3, label=f'Polynomial Trend (Degree {degree_3})', color='red')
plt.title('B - Polynomial Trend Estimation')
plt.xlabel('Year')
plt.ylabel('Average Close Price')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Print Linear Trend Equation
print(f"Linear Trend Equation: y = {regressor.coef_[0]:.2f} * x + {regressor.intercept_:.2f}")

# Print Polynomial Equations
print("Polynomial Trend Equation (Degree 2): y = {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly.coef_[2], regressor_poly.coef_[1], regressor_poly.intercept_))
print("Polynomial Trend Equation (Degree 3): y = {:.2f} * x^3 + {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly_3.coef_[3], regressor_poly_3.coef_[2], regressor_poly_3.coef_[1], regressor_poly_3.intercept_))
```

### OUTPUT:

A - LINEAR TREND ESTIMATION:
![image](https://github.com/user-attachments/assets/c99e6f39-09b7-4378-9790-279c3d6071d4)

B- POLYNOMIAL TREND ESTIMATION:
![image](https://github.com/user-attachments/assets/4a94fb5e-89f6-466c-9436-9f36e324bb6c)

### RESULT:
Thus the python program for linear and Polynomial Trend Estimation has been executed successfully.

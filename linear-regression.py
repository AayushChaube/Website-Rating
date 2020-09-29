# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:41:18 2019

@author: Aayush Chaube
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

data=pd.read_excel("website-ratings-data.xlsx") #Reading data
print(data.head())
print(data.describe())

plt.figure(figsize=(12, 6))
plt.scatter(data['User'], data['ratings'], c='blue')

plt.xlabel("USERS")
plt.ylabel("Ratings (out of 5)")
plt.show()

#now creating linear approximation
X=data['User'].values.reshape(-1, 1)
y=data['ratings'].values.reshape(-1, 1)
reg=LinearRegression()
reg.fit(X, y)
#reg.coef_ calculates slope; reg.intercept_ calculates intercept
print("The linear model is: Y = {:.5}X + {:.5}".format(reg.coef_[0][0], reg.intercept_[0]))

#now creating prediction
predictions=reg.predict(X)
plt.figure(figsize=(12, 6))
plt.scatter(data['User'], data['ratings'], c='black')
plt.plot(data['User'], predictions, c='red', linewidth=2)
plt.xlabel("USERS")
plt.ylabel("Ratings (Out of 5)")
plt.show()

#now assesing efficiency using R-Squared model
X=data['User']
y=data['ratings']
X2=sm.add_constant(X)
#Ordimary Latest Squares is the simplest and most common estimator in which the two\(\beta\)s are chosen to minimize the square of the distance between the predicted values and the actual values.
est=sm.OLS(y, X2)
est2=est.fit()
print(est2.summary())

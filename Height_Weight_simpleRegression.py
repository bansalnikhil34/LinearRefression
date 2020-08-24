#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:17:13 2020

@author: SIRIONLABS\nikhil.bansal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Weight.csv')

# divide datset into x and y

x = df.iloc[:,:-1]
y= df.iloc[:,-1]

# divide datset into training and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=2/3)

# implement linear regression model

from sklearn.linear_model import LinearRegression

simpleRegression = LinearRegression()

simpleRegression.fit(x_train,y_train)

y_predict = simpleRegression.predict(x_test)

y_predict_val = simpleRegression.predict([[2]])

# show the graph

plt.scatter(x_train,y_train)
plt.plot(x_train,simpleRegression.predict(x_train))
plt.show()



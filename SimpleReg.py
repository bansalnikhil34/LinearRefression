#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:54:01 2020

@author: SIRIONLABS\nikhil.bansal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_csv("Salary_Data.csv")

# divide dataset into x and y

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

# divide the dataset into training and test dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# Implement our classifiers based on simple linear regresion

from sklearn.linear_model import LinearRegression

simpleRegression = LinearRegression()
simpleRegression.fit(x_train,y_train) 

y_predict = simpleRegression.predict(x_test)


# need to provide value in 2d array
y_predict_val = simpleRegression.predict([[12]])

# implement the graph for simple linear regression

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,simpleRegression.predict(x_train))
plt.show()

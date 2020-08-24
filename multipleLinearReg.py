#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:13:47 2020

@author: SIRIONLABS\nikhil.bansal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('50_Startups.csv')

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

# convert categorical data

states=pd.get_dummies(x['State'],drop_first=True)

x=x.drop('State',axis=1)

# concat states with x

x=pd.concat([x,states],axis=1)

# spliting the data into trining and test data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

# Linear Regression for multiple variables 

from sklearn.linear_model import LinearRegression

multipleRegression=LinearRegression()

multipleRegression.fit(x_train,y_train)

# prediction for x_test data

y_predict = multipleRegression.predict(x_test)

# check if the model is good or not

from sklearn.metrics import r2_score

score = r2_score(y_test,y_predict)
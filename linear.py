#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:10:55 2020

@author: SIRIONLABS\nikhil.bansal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

df=load_boston()

dataset = pd.DataFrame(df.data)
dataset.columns=df.feature_names

dataset['Price']=df.target


x=dataset.iloc[:,:-1]

y=dataset.iloc[:,-1]


from sklearn.linear_model import LinearRegression

linearReg1 = LinearRegression()
# cross_val_score is actually used to do cross validation

from sklearn.model_selection import cross_val_score

# calculate mean squared error

mse=cross_val_score(linearReg1,x,y,scoring='neg_mean_squared_error',cv=5)

mean_mse = np.mean(mse)



# prediction

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

linearReg1.fit(x_train,y_train)

y_predict_linear = linearReg1.predict(x_test)

# sudhan comment
# plot
sns.distplot(y_test-y_predict_linear)
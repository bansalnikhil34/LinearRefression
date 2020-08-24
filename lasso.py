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

# LASO Regression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso=Lasso()

parameters= {'alpha':[1e-15,1e-10,1e-8,1e-5,1e-3,1e-2,1,5,10,20,25,30,35,40,45,50,55,70,80,90,100]}

lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)

best_param_lasso = lasso_regressor.best_params_
best_score_lasso = lasso_regressor.best_score_


# prediction

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.8,random_state=0)

lasso_regressor.fit(x_train,y_train)

y_predict_lasso=lasso_regressor.predict(x_test)

# plot
sns.distplot(y_test-y_predict_lasso)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 09:37:21 2020

@author: alex_chiang
"""

##### Features Engineering ##### 

'''
Using past N days to predict future N days TWII
'''

import numpy as np
import pandas as pd

df = pd.read_pickle("/Users/alex_chiang/Documents/GitHub/An-example-for-Machine-Learning-to-Predict-TWII/Data_clean.pkl")
df1 = df.copy(deep=True)  

col_name = df.columns.tolist()
col_name2 = col_name.copy()

pdays = 5 # Past N days
fdays = 5 # Future N days

for i in col_name:
    for j in range(1,int(pdays)):
        k = i + "-" +str(j)
        col_name2.insert(col_name2.index(i)+j,k)
df = df.reindex(columns=col_name2)

for i in col_name:
    for j in range(1,int(pdays)):
        k = i + "-" +str(j)
        df[k]=df[i].rolling(j+1).mean()
         
df = df.drop(df.head(int(pdays)-1).index)
df['expect_TWII'] = df1['^TWII'].shift(-int(fdays))
df = df.drop(df.tail(int(fdays)).index)
df.isnull().values.any()

df.to_pickle("/Users/alex_chiang/Documents/GitHub/An-example-for-Machine-Learning-to-Predict-TWII/Data_ML.pkl")
#%%

##### Split Data #####

import numpy as np
import pandas as pd
import datetime

df = pd.read_pickle("/Users/alex_chiang/Documents/GitHub/An-example-for-Machine-Learning-to-Predict-TWII/Data_ML.pkl")

split_date= datetime.datetime(2019, 1, 1)

X_train = df.loc[:split_date,df.columns[:-1]]
y_train = df.loc[:split_date,df.columns[-1]]
X_test = df.loc[split_date:,df.columns[:-1]]
y_test = df.loc[split_date:,df.columns[-1]]
#%%

##### Modeling #####

# UDF in pipeline
# https://stackoverflow.com/questions/31259891/put-customized-functions-in-sklearn-pipeline

def trans_func1(X):  
    # all features
    X = X
    return X

def trans_func2(X):  
    # Besides TWII
    feature_target = [col for col in X.columns if "TWII" in col]
    X = X[X.columns.difference(feature_target)]
    return X
        
def trans_func3(X):  
    # Only TWII
    feature_target = [col for col in X.columns if "TWII" in col]
    X = X[feature_target]
    return X

# Transform function
from sklearn.preprocessing import FunctionTransformer
transformer1 = FunctionTransformer(trans_func1, validate=False)
transformer2 = FunctionTransformer(trans_func2, validate=False)
transformer3 = FunctionTransformer(trans_func3, validate=False)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# pipeline1~3 Choosing Features
# pipeline4 Principal Component Analysis
# pipeline5 MinMaxScaler or StandardScaler
# pipeline6 Lasso Regression
pipeline1 = ('trans_func1', "passthrough")
pipeline2 = ('trans_func2', "passthrough")
pipeline3 = ('trans_func3', "passthrough")
pipeline4 = ("dimensional_reduction", "passthrough") 
pipeline5 = ("scaler", "passthrough")
pipeline6 = ("regressor", "passthrough")
estimator = Pipeline([pipeline1, pipeline2, pipeline3, pipeline4, pipeline5, pipeline6])

estimator.get_params().keys() # find params
random_state = 123457

param_grid1 = {
    'trans_func1': [None],
    'trans_func2': [None],
    'trans_func3': [transformer3],
    "dimensional_reduction": [None],
    "scaler": [StandardScaler(), MinMaxScaler()],
    "regressor": [Lasso(random_state=random_state)],
    "regressor__alpha": np.linspace(0.10, 0.2, 11, endpoint=True), 
    }

param_grid2 = {
    'trans_func1': [None],
    'trans_func2': [transformer2],
    'trans_func3': [None],
    "dimensional_reduction": [PCA()],
    "dimensional_reduction__n_components": [3, 5], 
    "scaler": [StandardScaler(), MinMaxScaler()],
    "regressor": [Lasso(random_state=random_state)],
    "regressor__alpha": np.linspace(0.10, 0.2, 11, endpoint=True), 
    }

param_grid3 = {
    'trans_func1': [transformer1],
    'trans_func2': [None],
    'trans_func3': [None],
    "dimensional_reduction": [PCA()],
    "dimensional_reduction__n_components": [3, 5], 
    "scaler": [StandardScaler(), MinMaxScaler()],
    "regressor": [Lasso(random_state=random_state)],
    "regressor__alpha": np.linspace(0.10, 0.2, 11, endpoint=True), 
    }

param_grid = [param_grid1, param_grid2, param_grid3]

scoring = "neg_mean_squared_error"
n_splits = 4
cv = TimeSeriesSplit(n_splits=n_splits)

gridsearchCV = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid,
                            scoring=scoring, 
                            cv=cv,
                            return_train_score=True, 
                            verbose=1,
                            n_jobs=1)

gridsearchCV.fit(X_train, y_train)
gridsearchCV.score(X_test, y_test)
#%%

##### Result #####

# GridSearchCV result
cv_results = gridsearchCV.cv_results_
sorted(cv_results.keys())
cv_results_df = pd.DataFrame(cv_results)

# Best model's score
rank = cv_results_df.sort_values('rank_test_score')
mean_train_score = rank['mean_train_score'][0]
mean_validation_score = rank['mean_test_score'][0]
print('mean train score:', mean_train_score)
print('mean validation score:', mean_validation_score)

# Get best model
best_estimator = gridsearchCV.best_estimator_
print("best_estimator =", best_estimator)

# Show in pictures
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)
y_train_df['pred'] = best_estimator.predict(X_train)
y_test_df['pred'] = best_estimator.predict(X_test)

import matplotlib.pyplot as plt 
fontsize = 14
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize

def plot_target_pred(y, y_pred, target_future, title):
    plt.figure() # draw the first one
    target_future_pred = target_future+"_pred"
    plt.plot(y, "b-", label=target_future)
    plt.plot(y_pred, "r-.", label=target_future_pred)
    plt.legend() # draw together
    plt.title(title)
    plt.figure() # draw the other one
    target_future_pred = target_future+"_pred"
    plt.plot(y, y_pred, ".")
    plt.xlabel(target_future)
    plt.ylabel(target_future_pred)
    plt.title(title)
    plt.show()
    
plot_target_pred(y_train_df['expect_TWII'],
                 y_train_df['pred'],
                 'TWII_5dayAfter', 
                 "TrainPredict")

plot_target_pred(y_test_df['expect_TWII'],
                 y_test_df['pred'],
                 'TWII_5dayAfter', 
                 "TestPredict")

# Compare score with alpha
from sklearn.metrics import mean_squared_error
def score_α(Alpha):
    ooo = best_estimator
    best_regressor = best_estimator.steps[5][1]
    best_regressor.set_params(alpha = Alpha) 
    ooo.fit(X_train,y_train)
    # print(ooo.coef_)  
    Train_y_pred = ooo.predict(X_train) 
    Test_y_pred = ooo.predict(X_test)     
    # Train
    MSE_train = mean_squared_error(y_train, Train_y_pred)
    Train_y = pd.DataFrame(y_train)
    Train_y['α='+str(Alpha)] = Train_y_pred
    #Train_y.plot(color=['turquoise','royalblue'],grid = True,title = 'α='+str(Alpha)+' - Train')
    # TEST
    MSE_test = mean_squared_error(y_test, Test_y_pred)
    Test_y = pd.DataFrame(y_test)
    Test_y['α='+str(Alpha)] = Test_y_pred
    #Test_y.plot(color=['turquoise','royalblue'],grid = True,title = 'α='+str(Alpha)+' - Test')
    return MSE_train,MSE_test

MSE_TRAIN = []
MSE_TEST = []
alphas = np.linspace(1,0.001,1000)
for a in alphas:
    MSE_TRAIN.append(score_α(a)[0])
    MSE_TEST.append(score_α(a)[1])

CompareAlpha = pd.DataFrame(MSE_TRAIN,index=alphas)
CompareAlpha.columns = ['MSE_TRAIN']
CompareAlpha['MSE_TEST'] = MSE_TEST
CompareAlpha.plot(secondary_y=['MSE_TEST'],logx = True,color=['turquoise','royalblue'],grid = True,title = 'Alpha').invert_xaxis()

score_α(0.01)
score_α(0.008)
score_α(0.005)
score_α(0.003)
score_α(0.001)
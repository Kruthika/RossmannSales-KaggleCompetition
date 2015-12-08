# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:16:36 2015

@author: Kruthika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#Predicting the sales using linear regression from test data
#Obtain sales, store data
trainingData = pd.read_csv('train.csv')
stores = pd.read_csv('store.csv')
stores.shape

#Obtain testing data
testingData = pd.read_csv('test.csv', sep=',', low_memory=False)

# Assume all stores are open and with sales greater than 0
trainingData.loc[trainingData.Open.isnull(), 'Open'] = 1
trainingData.loc[trainingData.Sales.isnull(), 'Sales'] = 1

#Fill in missing values similar to the training set
stores.CompetitionOpenSinceYear.fillna(2015, inplace=True)
stores.CompetitionOpenSinceMonth.fillna(8, inplace=True)
stores.Promo2SinceYear.fillna(2015, inplace=True)
stores.Promo2SinceWeek.fillna(26, inplace=True)
stores.PromoInterval.fillna('Jan,Apr,Jul,Oct',inplace=True )

#Merge train and test data with store data 
trainingData = pd.merge(trainingData,stores,on='Store')
testingData = pd.merge(trainingData,stores,on='Store')

#Defining a function to obtain required features
def processing(trainingData):
    trainingData['date'] = pd.to_datetime(trainingData['Date'])
    trainingData['year'] = pd.DatetimeIndex(trainingData['Date']).year
    trainingData['month'] = pd.DatetimeIndex(trainingData['Date']).month
    trainingData['DayOfWeek'] = pd.DatetimeIndex(trainingData['Date']).dayofweek
    trainingData['WeekOfYear'] = pd.DatetimeIndex(trainingData['Date']).weekofyear
    
    #Map each of the variables to a number
    trainingData['StoreType'] = trainingData.StoreType.map({'a':0, 'b':1, 'c':2, 'd':3})
    trainingData['Assortment'] = trainingData.Assortment.map({'a':0, 'b':1, 'c':2})
    trainingData['StateHoliday'] =trainingData.StateHoliday.map({'a':0, 'b':1, 'c':2, '0':0})
    trainingData['PromoInterval'] = trainingData.PromoInterval.map({'Jan, Apr,Jul, Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3})
    
    # Calculate time competition open time in months
    trainingData['CompetitionOpen'] = 12 * (trainingData.year - trainingData.CompetitionOpenSinceYear) + (trainingData.month - trainingData.CompetitionOpenSinceMonth)
    # Promo open time in month
    trainingData['PromoOpen'] = 12 * (trainingData.year - trainingData.Promo2SinceYear) + (trainingData.WeekOfYear - trainingData.Promo2SinceWeek) / 4.0
    
    return trainingData
    
#Obtaining training and testing Data
trainingData = processing(trainingData)
testingData = processing(testingData)

# Check for same columns in test data as training
for col in trainingData.column:
    if col not in testingData.column:
        testingData[col] = np.zeros(testingData.shape[0])
        
# Model the data using linear regression and predict the sales
from sklearn.linear_model import LinearRegression

feature_cols = ['CompetitionDistance', 'StoreType', 'Assortment','StateHoliday','Promo', 'Promo2', 'SchoolHoliday', 'DayOfWeek', 'month', 'WeekOfYear', 'CompetitionOpen', 'PromoOpen']

X = trainingData[feature_cols]
y = trainingData.Sales

lm = LinearRegression()
lm.fit(X, y)

X_test = trainingData[features_cols]
# make predictions for testing set
y_pred = lm.predict(X_test)
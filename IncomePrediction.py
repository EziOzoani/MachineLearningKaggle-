#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 17:34:56 2019

@author: ezi
"""

import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
%matplotlib inline

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# Loading dataset

data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv') 

data

# ------------------------------------------------------------------------------------------------------

# Dataset analysis

data[data.isnull().any(axis=1)].head()

import numpy as np
np.sum(data.isnull().any(axis=1))

# Filling the values of NaN with 0

data=data.dropna(axis = 0, how ='any') 

import numpy as np
np.sum(data.isnull().any(axis=1))

data.isnull().any(axis=0)

data.info()

data.describe()

# Looking at the all columns

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas
neg = data
neg_string = []
for t in neg:
    neg_string.append(t)
neg_string = pandas.Series(neg_string).str.cat(sep=' ')


wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

data.head(2)

# Encoding Gender, Country, Profession, University Degree and Hair Color

from sklearn import preprocessing

ge= preprocessing.LabelEncoder()
encge=ge.fit_transform(data['Gender'])
data['Gender'] = encge

co= preprocessing.LabelEncoder()
encco=co.fit_transform(data['Country'])
data['Country'] = encco

pr= preprocessing.LabelEncoder()
encpr=pr.fit_transform(data['Profession'])
data['Profession'] = encpr

uni= preprocessing.LabelEncoder()
encuni=uni.fit_transform(data['University Degree'])
data['University Degree'] = encuni

ha= preprocessing.LabelEncoder()
encha=ha.fit_transform(data['Hair Color'])
data['Hair Color'] = encha

# Heatmap for looking at the values of all columns

plt.figure(figsize = (15,15))
sns.heatmap(data = data.corr(), annot=True, linewidths=.3, cmap='RdBu')
plt.show()

data.describe()

# Features Distribution graphs

data.hist(figsize=(15,12),bins = 20, color="#107009AA")
plt.title("Features Distribution")
plt.show()

# By getting features and Class

y=data['Income in EUR']
X=data.drop(columns=['Income in EUR','Instance'])

# LinearRegression
#from sklearn import linear_model
#LR=linear_model.LinearRegression()
#LR= LR.fit(X , y)
#LR

# Fitting Decision Tree Regression to the dataset
#from sklearn.tree import DecisionTreeRegressor
#LR = DecisionTreeRegressor(random_state = 0)
#LR=LR.fit(X, y)
#LR

# Fitting SupportVectorRegression to the dataset
#from sklearn.svm import SVR
#LR = SVR(kernel = 'rbf')
#LR=LR.fit(X, y)
#LR

# Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor
LR = RandomForestRegressor(n_estimators = 10, random_state = 0)
LR= LR.fit(X, y)
LR

# getting the data for predictions 

import pandas as pd

DataS=pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
DataS=DataS.drop(columns=['Instance','Income'])
DataS=DataS.dropna(axis = 0, how ='any') 
DataS

from sklearn import preprocessing

ge= preprocessing.LabelEncoder()
encge=ge.fit_transform(DataS['Gender'])
DataS['Gender'] = encge

co= preprocessing.LabelEncoder()
encco=co.fit_transform(DataS['Country'])
DataS['Country'] = encco

pr= preprocessing.LabelEncoder()
encpr=pr.fit_transform(DataS['Profession'])
DataS['Profession'] = encpr

uni= preprocessing.LabelEncoder()
encuni=uni.fit_transform(DataS['University Degree'])
DataS['University Degree'] = encuni

ha= preprocessing.LabelEncoder()
encha=ha.fit_transform(DataS['Hair Color'])
DataS['Hair Color'] = encha

# Getting Predictions with Trained model


#pred=classifier.predict(DataS)

pred=LR.predict(DataS)

# Predictions

pred

# Saving Predictions in pandas

submission = pd.DataFrame()
submission["Action"]=pred

import pandas as pd

DataS=pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

DataS['Income']=submission["Action"]

DataS

# Saving with labels

DataS.to_csv('Predictions_tcd ml 2019-20 income prediction test (without labels)')

submission = pd.DataFrame()
submission["Income"]=pred

submission.index += 1

submission["Instance"] = submission.index

submission=submission[['Instance','Income']]

# Saving the Submission File

filename = 'tcd ml 2019-20 income prediction submission file.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:38:17 2023

@author: jeff
"""

# load packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score

# load the data file

df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2824 Big Data and Forcasting in Econ\Assignments\Assignment1\wine_data.csv')
print(df)

# summary statistics

print(df.describe())

# drop missing values

df = df.dropna()
print(df)
print(df.describe())

# plot each independent variable and check for outliers and how the variable is distributed. drop outliers if any. calculate correlation to quality. 

df.drop(df[df['chlorides']>0.2].index, inplace = True)
print(df)

plt.scatter(df['quality'],df['chlorides'])

sns.kdeplot(df['chlorides'])

print(df['quality'].corr(df['chlorides']))

# same process for fixed acidity

df.drop(df[df['fixed_acidity']<0].index, inplace = True)
print(df)

plt.scatter(df['quality'],df['fixed_acidity'])
plt.show()

sns.kdeplot(df['fixed_acidity'])

print(df['quality'].corr(df['fixed_acidity']))

# same process for volatile acidity

plt.scatter(df['quality'],df['volatile_acidity'])
plt.show()

df.drop(df[df['volatile_acidity']>1.2].index, inplace = True)
print(df)

sns.kdeplot(df['volatile_acidity'])

print(df['quality'].corr(df['volatile_acidity']))

# same process for citric acid

plt.scatter(df['quality'],df['citric_acid'])
plt.show()

sns.kdeplot(df['citric_acid'])

print(df['quality'].corr(df['citric_acid']))

# same process for residual sugar

plt.scatter(df['quality'],df['residual_sugar'])
plt.show()

df.drop(df[df['residual_sugar']>5.5].index, inplace = True)
print(df)

sns.kdeplot(df['residual_sugar'])

print(df['quality'].corr(df['residual_sugar']))

# same process for free sulfur dioxide

plt.scatter(df['quality'],df['free_sulfur_dioxide'])
plt.show()

df.drop(df[df['free_sulfur_dioxide']>50].index, inplace = True)
print(df)

sns.kdeplot(df['free_sulfur_dioxide'])

print(df['quality'].corr(df['free_sulfur_dioxide']))

# same process for total sulfur dioxide

plt.scatter(df['quality'],df['total_sulfur_dioxide'])
plt.show()

df.drop(df[df['total_sulfur_dioxide']>150].index, inplace = True)
print(df)

sns.kdeplot(df['total_sulfur_dioxide'])

print(df['quality'].corr(df['total_sulfur_dioxide']))

# same process for density 

plt.scatter(df['quality'],df['density'])
plt.show()

sns.kdeplot(df['density'])

print(df['quality'].corr(df['density']))

# same process for pH

df.drop(df[df['pH']>5].index, inplace = True)
print(df)

plt.scatter(df['quality'],df['pH'])
plt.show()

sns.kdeplot(df['pH'])

print(df['quality'].corr(df['pH']))

# same process for sulphates

df.drop(df[df['sulphates']>1.25].index, inplace = True)
print(df)

plt.scatter(df['quality'],df['sulphates'])
plt.show()

sns.kdeplot(df['sulphates'])

print(df['quality'].corr(df['sulphates']))

# same process for alcohol

df.drop(df[df['alcohol']>=14].index, inplace = True)
print(df)

plt.scatter(df['quality'],df['alcohol'])
plt.show()

sns.kdeplot(df['alcohol'])

print(df['quality'].corr(df['alcohol']))

# convert quality to a binary variable, where it is 1 if greater than 6, 0 otherwise

df['high_quality'] = np.where(df['quality'] > 6, 1, 0)

# save and export the cleaned data frame

df.to_csv('cleaned_df.csv')

# split dataframe into train/test dataset

features = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']
x = df.loc[:,features]
y = df.loc[:,['high_quality']]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, train_size=0.8)

# implement Gaussian naive bayes classifier 

model = GaussianNB()

# train the model

model.fit(xtrain, ytrain)

# predict the output

pred = model.predict(xtest)

# rate the model

model.score(x,y)

# create confusion matrix

mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Truth')
plt.ylabel('Predicted')

# implement Bernoulli naive bayes classifier

bmodel = BernoulliNB()

# train the model

bmodel.fit(xtrain, ytrain)

# predict the output

bpred = bmodel.predict(xtest)

# rate the model

bmodel.score(x,y)

# create confusion matrix

mat2 = confusion_matrix(bpred, ytest)
names = np.unique(bpred)
sns.heatmap(mat2, annot=True, fmt='d', cbar=False)
plt.xlabel('Truth')
plt.ylabel('Predicted')

# new attempt with less predictors

# split dataframe into train/test dataset

new_features = ['volatile_acidity','citric_acid','sulphates','alcohol']
x2 = df.loc[:,new_features]
y2 = df.loc[:,['high_quality']]

xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x2, y2, random_state=0, train_size=0.8)

# implement Gaussian naive bayes classifier 

model2 = GaussianNB()

# train the model

model2.fit(xtrain2, ytrain2)

# predict the output

pred2 = model2.predict(xtest2)

# rate the model

model2.score(x2,y2)

# create confusion matrix

mat2 = confusion_matrix(pred2, ytest2)
names2 = np.unique(pred2)
sns.heatmap(mat2, annot=True, fmt='d', cbar=False)
plt.xlabel('Truth')
plt.ylabel('Predicted')

# The first Gaussian model seems to perform the best. Now, it is time to cross validate the model by precision.

scores = cross_val_score(model, xtrain, ytrain, cv=10, scoring='precision')
print(scores)
meanScore = scores.mean()
print(meanScore * 100)

# Also, it is good to cross validate the negative predicted value

neg_scores = cross_val_score(model, xtrain, ytrain, cv=10, scoring=make_scorer(precision_score, pos_label=0))
print(neg_scores)
meanNegScore = neg_scores.mean()
print(meanNegScore * 100)

# To furthur test the robustness of our first Gaussian model, I will perform the same cross validation process again for Bernoulli model.

scores2 = cross_val_score(bmodel, xtrain, ytrain, cv=10, scoring='precision')
print(scores2)
meanScore2 = scores2.mean()
print(meanScore2 * 100)

neg_scores2 = cross_val_score(bmodel, xtrain, ytrain, cv=10, scoring=make_scorer(precision_score, pos_label=0))
print(neg_scores2)
meanNegScore2 = neg_scores2.mean()
print(meanScore * 100)
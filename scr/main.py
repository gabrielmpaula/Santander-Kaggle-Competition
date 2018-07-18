# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:42:54 2018

@author: Gabriel
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

original_train_df = pd.read_csv('../input/train.csv')
original_test_df = pd.read_csv('../input/test.csv')

original_train_df.head(5)

target = original_train_df['target']
train_df = original_train_df.drop(columns='target')
test_df = original_test_df.copy()
train_df = train_df.set_index('ID')

print('Train Dataset information:')
train_df.info()
print()
print('Test Dataset information:')
test_df.info()

sns.distplot(target)

bc_target, lam = boxcox(target)
print('Box-Cox lambda value: {}'.format('%.3f' % lam))
#plt.subplot(211)
sns.distplot(bc_target);
#plt.subplot(212)
#sns.distplot(inv_boxcox(bc_target,lam));

bc_target_df = pd.DataFrame(bc_target,columns=['target'])
plt.plot(range(bc_target_df.shape[0]),bc_target_df.sort_values(by='target',ascending=True))

X_train, X_test, y_train, y_test = train_test_split(train_df, bc_target, test_size=0.25, random_state=0)

seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=5, random_state=seed)

for alpha in (0.0001, 0.001, 0.01, 0.1, 1, 3, 5, 6, 7, 8, 9, 10, 50, 100):
    ridge_model = Ridge(alpha = alpha)
    ridge_model.fit(X_train, y_train)
    ridge_model_pred = inv_boxcox(ridge_model.predict(X_test),lam)
      
    print ('Alpha =', alpha)
    #print('Ridge Regression R squared Train:' '%.4f' % ridge_model.score(X_train, y_train))
    #print('Ridge Regression R squared Test:' '%.4f' % ridge_model.score(X_test, y_test))
    #results = cross_val_score(ridge_model, train_df, bc_target, cv=kfold, scoring='neg_mean_squared_error') 
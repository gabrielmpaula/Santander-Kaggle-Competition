
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import boxcox
from scipy.special import boxcox1p
from scipy.special import inv_boxcox
from scipy.special import inv_boxcox1p

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from keras import backend
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[2]:


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# In[3]:


def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(y-y0, 2)))


# In[4]:


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='mse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# In[ ]:


original_train_df = pd.read_csv('../input/train.csv')
original_test_df = pd.read_csv('../input/test.csv')


# In[ ]:


original_train_df.head(5)


# In[ ]:


target = original_train_df['target']
target_df = pd.DataFrame(target, columns=['target'])

train_df = original_train_df.drop(columns=['target'])
train_df = train_df.set_index('ID')
test_df = original_test_df.copy()
test_df = test_df.set_index('ID')
#train_df = train_df.set_index('ID')


# In[ ]:


print('Train Dataset shape:')
print(train_df.shape)
print()
print('Test Dataset shape:')
print(test_df.shape)


# In[ ]:


sns.distplot(target);


# In[ ]:


cols_with_unique_value = train_df.columns[train_df.nunique()==1]
train_df_drop = train_df.drop(cols_with_unique_value, axis = 1)
test_df_drop = test_df.drop(cols_with_unique_value, axis = 1)
train_df_drop.shape


# In[ ]:


i = 1
train_df_drop[columns[i]].values


# In[ ]:


cols_to_remove = []
columns = train_df_drop.columns
for i in range(len(columns)-1):
    col_value_test = train_df_drop[columns[i]].values
    for j in range(i+1, len(columns)):
        if np.array_equal(col_value_test, train_df_drop[columns[j]]):
            cols_to_remove.append(columns[j])
            


# In[ ]:


# lam = 0.15
# bc_train_df_drop = boxcox1p(train_df_drop, lam)


# In[ ]:


log_target = np.log1p(target)
log_target_df = pd.DataFrame(log_target,columns=['target'])
#print('Box-Cox lambda value: {}'.format('%.3f' % lam))

# lam = 0.11
# bc_target = boxcox1p(target_df, lam)
# bc_target_df = pd.DataFrame(bc_target,columns=['target'])

# plt.subplot(211)
sns.distplot(log_target_df);
# plt.subplot(212)
# sns.distplot(bc_target);


# In[ ]:


#plt.plot(range(bc_target_df.shape[0]),bc_target_df.sort_values(by='target',ascending=True));


# In[ ]:


scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(test_df_drop)
train_df_scaled = pd.DataFrame(train_df_scaled)
train_df_scaled.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df_drop, log_target_df, test_size=0.25, random_state=0)


# In[ ]:


print(train_df_drop.shape)
print(train_df.shape)


# In[ ]:


seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
#help(kfold.split)


# In[ ]:


# for alpha in [2.5e17, 5e17, 8e17, 1e18, 5e18, 1e19]:
#     ridge_model = Ridge(alpha = alpha)    
#     print('Training for alpha:', alpha)
#     total_error = 0
#     for train_indices, test_indices in kfold.split(train_df):
        
#         X_train_subset = train_df_drop.iloc[train_indices,:]
#         y_train_subset = log_target_df.iloc[train_indices]

#         X_test_subset = train_df_drop.iloc[test_indices,:]
#         y_test_subset = log_target_df.iloc[test_indices]

#         ridge_model.fit(X_train_subset, y_train_subset)
#         model_pred = ridge_model.predict(X_test_subset)

#         model_error = np.sqrt(mean_squared_error(model_pred, y_test_subset))
#         print('RMSLE:', model_error)
#         total_error += model_error
        
#     print('Alpha',alpha,'Average RMSLE:', (total_error/5),'\n')
# print('FINISHED')


# In[ ]:


# for alpha in [9e4, 1e5, 5e5, 1e6]:
#     lasso_model = Lasso(alpha = alpha)    
#     print('Training for alpha:', alpha)
#     total_error = 0
#     for train_indices, test_indices in kfold.split(train_df):
        
#         X_train_subset = train_df_drop.iloc[train_indices,:]
#         y_train_subset = log_target_df.iloc[train_indices]

#         X_test_subset = train_df_drop.iloc[test_indices,:]
#         y_test_subset = log_target_df.iloc[test_indices]

#         lasso_model.fit(X_train_subset, y_train_subset)
#         model_pred = lasso_model.predict(X_test_subset)

#         model_error = np.sqrt(mean_squared_error(model_pred, y_test_subset))
#         print('RMSLE:', model_error)
#         total_error += model_error
        
#     print('Alpha',alpha,'Average RMSLE:', (total_error/5),'\n')
# print('FINISHED')


# In[ ]:


# xgb_model = XGBRegressor()
# learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# n_estimators = [100, 150, 200, 500]
# max_depth = [2, 4, 6, 8]
# param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
# kfold = KFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(xgb_model, param_grid, scoring="neg_mean_squared_log_error", n_jobs=1, cv=kfold, verbose=True)
# grid_result = grid_search.fit(train_df_drop, log_target_df)


# In[ ]:


learning_rate = 0.1
max_depth = 8
n_estimators = 400
early_stopping_rounds = 50
cv = 5
xgb_model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
eval_set = [(X_test, y_test)]


# In[ ]:


xgb_model = XGBRegressor(
 learning_rate =0.02,
 n_estimators=2000,
 max_depth=32,
 min_child_weight=57,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 nthread=-1,
 scale_pos_weight=1,
 seed=27)


# In[ ]:


# results = cross_val_score(xgb_model, train_df_drop, log_target_df, cv=kfold, n_jobs=1, scoring='neg_mean_squared_error', verbose=True)
# results = np.sqrt( np.abs( results ) ) #MSLE TO RMSLE


# In[ ]:


# print(np.sqrt(np.abs(results)))


# In[ ]:


xgtrain = xgb.DMatrix(train_df_drop, log_target_df)
xgb_param = xgb_model.get_xgb_params()


# In[ ]:


results = xgb.cv(xgb_param, xgtrain, num_boost_round=n_estimators, nfold=cv, metrics='rmse', early_stopping_rounds=early_stopping_rounds, verbose_eval=20)


# In[ ]:


xgb_model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric="rmse", eval_set=eval_set, verbose=50)


# In[ ]:


xgb_model.fit(train_df_drop, log_target_df, eval_metric="rmse", verbose=True)


# In[ ]:


xgb_pred = xgb_model.predict(test_df_drop)
#model_error = np.sqrt(mean_squared_error(xgb_pred, y_test))
#results = cross_val_score(xgb_model, train_df, log_target_df, cv=kfold, scoring=rmse)
#print(model_error)


# In[ ]:


xgb_pred_df = pd.DataFrame(test_df_drop.index.values, columns=["ID"])
xgb_pred_df['target'] = np.expm1(xgb_pred)


# In[ ]:


xgb_pred_df.head()


# In[ ]:


with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(xgb_pred_df)


# In[ ]:


xgb_pred_df.to_csv('prediction.csv')


# XGBoost (Log Target + Drop): 1.58336621676

# In[ ]:


results


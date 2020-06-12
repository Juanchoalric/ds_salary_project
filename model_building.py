# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:04:59 2020

@author: juan.alric
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('eda_data.csv')

# Choose relevant columns

df_model = df[['avg_salary','Rating', 'Size', 'Type of ownership', 'Industry', 
               'Sector', 'Revenue', 'competitors_count', 'hourly', 
               'employer_provided', 'job_state', 'same_state', 'age', 
               'python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'job_simp', 
               'seniority','desc_len']]

df_dum = pd.get_dummies(df_model)

X = df_dum.drop('avg_salary', axis=1)

y = df_dum['avg_salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                    test_size=0.2)

# Multiple Linear Regression
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
print(model.fit().summary())

lm = LinearRegression()
lm.fit(X_train, y_train)

print(np.mean(cross_val_score(
    lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# Get dummy data for the categorical columns
# train test splits

# Lasso Regression
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train, y_train)
print(np.mean(cross_val_score(
    lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

alpha = []
error = []

for i in range(1,1000):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(
        np.mean(
            cross_val_score(lml,
                            X_train, 
                            y_train, 
                            scoring='neg_mean_absolute_error',
                            cv=3)))
    
plt.plot(alpha,error)
plt.show()

error = tuple(zip(alpha, error))

df_err = pd.DataFrame(error, columns = ['alpha', 'error'])

df_err[df_err.error == max(df_err.error)]
    
# Random Forest
rf = RandomForestRegressor()

print(np.mean(cross_val_score(
    rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
# Tune models GridsearchCV
parameters = {'n_estimators': range(10, 300, 10), 
              'criterion': ('mse', 'mae'), 
              'max_features': ('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

print('Finished')
print(gs.best_score_)
print(gs.best_estimator_)

# test ensembles

y_pred_lm = lm.predict(X_test)

y_pred_lml = lm_l.predict(X_test)


y_pred_rf = gs.best_estimator_.predict(X_test)


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred_lm)
mean_absolute_error(y_test, y_pred_lml)
mean_absolute_error(y_test, y_pred_rf)






# -*- coding: utf-8 -*-
"""
Feb 25

@author: Yoonha Kim
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

credit_card = pd.read_csv("/Users/yoonhakim/Desktop/UBC/Project/CreditCard_Fraud_Detection/Data/creditcard.csv")

# train test split 
from sklearn.model_selection import train_test_split

X = credit_card.drop('Class', axis =1)
y = credit_card.Class.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
import statsmodels.api as sm

logit = sm.Logit(y,X)
logit.fit().summary()

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score

sk_logit = LogisticRegression()
sk_logit.fit(X_train, y_train)

np.mean(cross_val_score(sk_logit,X_train,y_train)) # score value 0.99898

# Lasso regression 
lasso = Lasso()
lasso.fit(X_train,y_train)
np.mean(cross_val_score(lasso,X_train,y_train)) # score value 0.001485

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lasso_i = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lasso_i,X_train,y_train)) )
    
plt.plot(alpha,error)

# figure is not informative to get better alpha value

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train,y_train)
np.mean(cross_val_score(ridge,X_train,y_train)) #score value 0.55239

alpha_ridge = []
error_ridge = []

for i in range(1,100):
    alpha_ridge.append(i/100)
    ridge_i = Ridge(alpha=(i/100))
    error_ridge.append(np.mean(cross_val_score(ridge_i,X_train,y_train)) )
    
plt.plot(alpha_ridge,error_ridge)
# figure shows that alpha = 1 (default value) is the best

# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
#np.mean(cross_val_score(gb,X_train,y_train)) 

# tune models GridsearchCV 
#from sklearn.model_selection import GridSearchCV
#parameters = {'loss':('deviance','exponential'),'n_estimators':range(10,300,50)}

#gs = GridSearchCV(gb,parameters)
#gs.fit(X_train,y_train)
# since Grid Search takes too much time due to large size of data, I just used suggested parameters

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
gb.score(X_test, y_test) # score is 0.999

# test models
tpred_logit = sk_logit.predict(X_test)
tpred_lasso= lasso.predict(X_test)
tpred_ridge= ridge.predict(X_test)
tpred_gb = gb.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_logit) #0.001369
mean_absolute_error(y_test,tpred_lasso) #0.0034
mean_absolute_error(y_test,tpred_ridge) #0.003439
mean_absolute_error(y_test,tpred_gb) #0.00094 

mean_absolute_error(y_test,(tpred_logit+tpred_gb)/2) # 0.00115
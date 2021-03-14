# -*- coding: utf-8 -*-
"""
Feb 25

@author: Yoonha Kim
"""
import pandas as pd
import numpy as np

credit_card = pd.read_csv("/Users/yoonhakim/Desktop/UBC/Project/Credit_Card_Fraud_Detection/Data/creditcard.csv")

# train test split 
from sklearn.model_selection import train_test_split

X = credit_card.drop('Class', axis =1)
y = credit_card.Class.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
import statsmodels.api as sm

logit = sm.Logit(y,X)
logit.fit().summary()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sk_logit = LogisticRegression()
sk_logit.fit(X_train, y_train)

np.mean(cross_val_score(sk_logit,X_train,y_train)) # score value 0.99898

 
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

# test models; caculate accuracy
tpred_logit = sk_logit.predict(X_test)
tpred_gb = gb.predict(X_test)

from sklearn.metrics import confusion_matrix
true_neg_lg, false_pos_lg, false_neg_lg, true_pos_lg = confusion_matrix(y_test, tpred_logit).ravel()

#accuracy_score 
acc_lg = (true_neg_lg+true_pos_lg)/((true_neg_lg+true_pos_lg)+false_pos_lg+false_neg_lg) # 0.9986

#or we can use this:

from sklearn.metrics import accuracy_score
accuracy_score(y_test,tpred_logit) #0.9986306660580738
accuracy_score(y_test,tpred_gb) #0.9990519995786665

# Productionization

import pickle
pickl = {'model': gb}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
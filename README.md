# Credit Card Fraud Detection

- Created a tool that detects Credit Card Fraud using R
- Data used can be found here: https://www.kaggle.com/mlg-ulb/creditcardfraud


## Explanatory Data Analysis

I looked at the histograms of data, the correlation between predictor variables, and the pivot table. Below are a few highlights.
![alt text](https://github.com/yoonhaK/CreditCard_Fraud_Detection/blob/main/Histogram.png)
![alt text](https://github.com/yoonhaK/CreditCard_Fraud_Detection/blob/main/Correlation%20Matrix.png)

## Model building
I tried four different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret.

Four different models I tried:

Logistic Regression – Because the y value we want to predict is either 1(fraud) or 0 (non-fraud), I thought a logistic regression would be effective
Lasso Regression – Because we have lots of predictor variables, I was wondering if there are non-effective predictor variables.
Ridge Regression - Because we have lots of predictor variables, I was wodering if shrinkage would be effective
Gradient Boosting – I thought that this would be a good fit since gradient boosting is used to perform classification and regression tasks.

## Model Performance
Logistic Regression - 0.001369
Lasso Regression - 0.0034
Ridge Regression - 0.003439
Gradient Boosting - 0.00094

Since MAE of Lasso is not improved when it is compared to ridge, it indicates that all of the predictors we have are effective for prediction of fraud.
Gradient Boosting outperforms Logistic Regression while Logistic Regression works really well too.

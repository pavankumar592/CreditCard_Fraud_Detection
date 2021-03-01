# Credit Card Fraud Detection

- Created a tool that detects Credit Card Fraud using R and Python
- Data used can be found here: https://www.kaggle.com/mlg-ulb/creditcardfraud

- Used Python version: 3.7
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
- For Web Framework Requirements: pip install -r requirements.txt
- Flask Productionization: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2


## Explanatory Data Analysis

I looked at the histograms of data, the correlation between predictor variables, and the pivot table. Below are a few highlights.
<img src="https://github.com/yoonhaK/CreditCard_Fraud_Detection/blob/main/Histogram.png" width="400"/>
<img src="https://github.com/yoonhaK/CreditCard_Fraud_Detection/blob/main/Correlation%20Matrix.png" width="400"/>
<img src="https://github.com/yoonhaK/CreditCard_Fraud_Detection/blob/main/Pivot%20Table.png" width="600"/>

## Model building
I tried four different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret.

Four different models I tried:

- Logistic Regression – Because the y value we want to predict is either 1(fraud) or 0 (non-fraud), I thought a logistic regression would be effective.
- Lasso Regression – Because we have lots of predictor variables, I was wondering if there are non-effective predictor variables.
- Ridge Regression - Because we have lots of predictor variables, I was wodering if shrinkage would be effective. 
- Gradient Boosting – I thought that this would be a good fit since gradient boosting is used to perform classification and regression tasks.

## Model Performance
- Logistic Regression - 0.001369
- Lasso Regression - 0.0034
- Ridge Regression - 0.003439
- Gradient Boosting - 0.00094

Since MAE of Lasso is not improved when it is compared to ridge, it indicates that all of the predictors we have are effective for prediction of fraud.
Gradient Boosting outperforms Logistic Regression while Logistic Regression works really well too.

### Productionization
I built a flaskAPI endpoint that is hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from Credit Card Transaction and returns an estimation of fraud(1) or non-fraud(0)


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

bank_churn = pd.read_csv('Bank_Customer_Churn_Prediction.csv')
bank_churn.drop('customer_id',axis=1,inplace=True)


encoder = LabelEncoder()
categorical =['gender','country']
bank_churn[categorical] = bank_churn[categorical].apply(encoder.fit_transform)


scaler = MinMaxScaler(feature_range=(0,1))
X = bank_churn.drop('churn',axis=1)
y = bank_churn['churn']
X = scaler.fit_transform(X)


model = LogisticRegression()
model.fit(X,y)


pickle.dump(model, open("ml_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
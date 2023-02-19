import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("DATA.csv")
df.drop('id', axis=1, inplace=True)
df['Vehicle_Age'].replace({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2},inplace=True)
df['Vehicle_Damage'].replace({'Yes': 1, 'No': 0},inplace=True)
df['Gender'].replace({'Male': 1, 'Female': 0},inplace=True)

X= df.drop(['Response'],axis=1)
y= df['Response']

sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

logmodel = LogisticRegression(C=1)
logmodel.fit(X_scaled, y)

pickle.dump(logmodel, open("ml_model.pkl", "wb")) # wb >> writing in binary
pickle.dump(sc, open("scaler.pkl", "wb"))
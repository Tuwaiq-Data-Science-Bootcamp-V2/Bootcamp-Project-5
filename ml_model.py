import pickle
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import tree


dataset = pd.read_csv('iris.csv')

dataset = dataset.drop(['Id'], axis=1)

X = dataset.drop(['Species'], axis=1)
y = dataset['Species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


LabelEnc = preprocessing.LabelEncoder()
y = LabelEnc.fit_transform(y)


tree_model = tree.DecisionTreeClassifier(random_state = 0)
tree_model.fit(X_scaled, y)

pickle.dump(tree_model, open("tree_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
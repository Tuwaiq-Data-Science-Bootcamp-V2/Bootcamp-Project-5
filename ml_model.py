import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# read dataset
dataset = pd.read_csv('seattle-weather.csv', encoding='latin-1')

# LabelEncoder for the Weather
le = LabelEncoder()
dataset['Weather'] = le.fit_transform(dataset['weather'])

# drop columns
dataset = dataset.drop(['weather'],axis=1)
dataset = dataset.drop(['date'],axis=1)

# Split the dataset into train and test
X = dataset.drop(['Weather'], axis = 1)
y = dataset['Weather']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# pickle file
pickle.dump(classifier, open("ml_model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
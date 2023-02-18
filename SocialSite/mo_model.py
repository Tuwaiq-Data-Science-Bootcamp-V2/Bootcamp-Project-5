import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

dataset = pd.read_csv('Social_Network_Ads.csv')
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender','Age','EstimatedSalary']]
y = dataset['Purchased']


scaler = StandardScaler()
X = scaler.fit_transform(X)

lr = LogisticRegression()
lr.fit(X,y)


pickle.dump(lr, open("ml_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

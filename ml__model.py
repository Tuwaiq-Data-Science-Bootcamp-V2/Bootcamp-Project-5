import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Job_Placement_Data.csv')
X = dataset.drop(columns='status',axis=1)
y = dataset['status']

d=pd.DataFrame(y.values,columns=['status'])
ytrain= d.apply(LabelEncoder().fit_transform)
y=ytrain

d=X[['gender','hsc_subject','undergrad_degree','work_experience','specialisation']]
xtrain= d.apply(LabelEncoder().fit_transform)
X[['gender','hsc_subject','undergrad_degree','work_experience','specialisation']]=xtrain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=23)
model = RandomForestClassifier()
model.fit(X,y)


pickle.dump(model, open("ml__model.pymodel.pkl", "wb"))

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv('Job_Placement_Data.csv')

# Here User ID is not suitable to predict the results,so we are ignore this coloumn
data = data[['gender','degree_percentage','work_experience','emp_test_percentage','status']]

# Here User ID is not suitable to predict the results,so we are ignore this coloumn
data = data[['gender','degree_percentage','work_experience','emp_test_percentage','status']]

# select all categorical variables
dataset_categorical = data.select_dtypes(include=['object'])

label_encoder = LabelEncoder()
dataset_categorical = dataset_categorical.apply(label_encoder.fit_transform)
dataset_categorical.head()

df = data.drop(dataset_categorical.columns, axis=1)# drop categorical
df = pd.concat([df, dataset_categorical], axis=1)# concat for dataframe

# convert target variable status to categorical
df['status'] = df['status'].astype('category')
df['status'].head()

X = df.drop(columns='status',axis=1)
y = df['status']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train,y_train)
predict = log_model.predict(X_test)

pickle.dump(log_model, open("ml_model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))


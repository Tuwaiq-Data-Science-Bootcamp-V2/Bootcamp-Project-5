import warnings
warnings.filterwarnings("ignore")





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd


dataset = pd.read_csv('Social_Network_Ads.csv')
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender','Age','EstimatedSalary']]


y = dataset['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
lm = LogisticRegression(random_state=0)
lm.fit(X_train,y_train)
from sklearn.metrics import classification_report

scalar = StandardScaler()
lr = LogisticRegression()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
lr.fit(X_train , y_train)
predict= lr.predict(X_test)
print(classification_report(y_test, predict))
import pickle
pickle.dump(lm, open("lm.pkl", "wb"))
pickle.dump(lr, open("lr.pkl", "wb"))
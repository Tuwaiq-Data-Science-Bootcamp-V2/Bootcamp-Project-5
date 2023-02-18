import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Social_Network_Ads.csv')

X= dataset.drop(['User ID','Gender','Purchased'],axis='columns')
Y= dataset[['Purchased']]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)
pickle.dump(logreg, open("ml_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
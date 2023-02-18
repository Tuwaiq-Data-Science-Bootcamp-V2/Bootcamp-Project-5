import pickle
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Hobby_Data.csv')

label_encoder= preprocessing.LabelEncoder()
# Encode labels that have yes/no values
df['Projects']= label_encoder.fit_transform(df['Projects'])
df['Act_sprt']= label_encoder.fit_transform(df['Act_sprt'])
df['Olympiad_Participation']= label_encoder.fit_transform(df['Olympiad_Participation'])
df['Scholarship']= label_encoder.fit_transform(df['Scholarship'])
df['Medals']= label_encoder.fit_transform(df['Medals'])
df['Career_sprt']= label_encoder.fit_transform(df['Career_sprt'])
df['School']= label_encoder.fit_transform(df['School'])
df['Fant_arts']= label_encoder.fit_transform(df['Fant_arts'])
df.drop('Won_arts', axis=1, inplace=True)
df.drop('Fav_sub', axis=1, inplace=True)


X = df.drop(labels='Predicted Hobby', axis=1)
y = df['Predicted Hobby']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=.30)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pickle.dump(clf, open("ml_model.pkl", "wb"))
pickle.dump(label_encoder, open("labelEnc.pkl", "wb"))

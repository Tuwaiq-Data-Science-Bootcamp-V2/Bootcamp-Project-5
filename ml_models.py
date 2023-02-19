# Importing the required libraries
# To ignore warnings
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# read the data
df = pd.read_csv('Hobby_Data.csv')

label_encoder = preprocessing.LabelEncoder()

# Encode labels that have yes/no values
df['Projects']= label_encoder.fit_transform(df['Projects'])
df['Act_sprt']= label_encoder.fit_transform(df['Act_sprt'])
df['Olympiad_Participation']= label_encoder.fit_transform(df['Olympiad_Participation'])
df['Scholarship']= label_encoder.fit_transform(df['Scholarship'])
df['Medals']= label_encoder.fit_transform(df['Medals'])
df['Career_sprt']= label_encoder.fit_transform(df['Career_sprt'])
df['School']= label_encoder.fit_transform(df['School'])
df['Fant_arts']= label_encoder.fit_transform(df['Fant_arts'])

# Encoding Won_arts column which has different values with new encoder object
le2 = preprocessing.LabelEncoder()
df['Won_arts']= le2.fit_transform(df['Won_arts'])

# Encoding Fav_sub column which has different values with new encoder object
le3 = preprocessing.LabelEncoder()
df['Fav_sub']= le3.fit_transform(df['Fav_sub'])


# Encode the last column Predicted Hobby with one Hot Encoder
encoder = OneHotEncoder(sparse=False)
df1 = encoder.fit_transform(df[['Predicted Hobby']])

# Merge df1 with df
lr_df = pd.concat([df,pd.DataFrame(df1)],axis=1)

# Rename new one hot encoder columns
lr_df.rename(columns={0: "Academics", 1: "Sports", 2: "Arts"},inplace=True,)

X = lr_df.drop(labels=['Predicted Hobby','Academics','Sports','Arts'], axis=1)
y1 = lr_df['Academics'] # y for academics
y2 = lr_df['Sports'] # y for sports
y3 = lr_df['Arts'] # y for Arts
lr = LogisticRegression()
lr.fit(X,y1)

y = df['Predicted Hobby']
# Create and train the model for sports
lrs = LogisticRegression()
lrs.fit(X,y2)

# Create and train the model for Arts
lra = LogisticRegression()
lra.fit(X,y3)

model = DecisionTreeClassifier()
model.fit(X,y)

pickle.dump(lra, open("lra.pkl", "wb"))
pickle.dump(lr, open("lr.pkl", "wb"))
pickle.dump(lrs, open("lrs.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

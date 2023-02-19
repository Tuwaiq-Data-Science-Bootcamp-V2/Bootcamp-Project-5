# Load libraries
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('instagram_users.csv',encoding='latin-1')

df=df.rename(columns=lambda x: x.strip().lower())
df['real_fake']=df['real_fake'].apply(lambda x:0 if x=='fake' else 0)
featuer=['num_posts','num_following','num_followers','biography_length','picture_availability','link_availability','average_caption_length']

X=df[featuer]
y=df['real_fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print('the model name :', rfc ,end='\n' )
print('the model accuracy : ' ,metrics.accuracy_score(y_test, rfc_pred)*100)
print('the model confusion matrix. : ',metrics.confusion_matrix(y_test,rfc_pred))

pickle.dump(rfc, open("ml_model.pkl", "wb"))

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




# In[5]:


df = pd.read_csv('PersonalityTest.csv')


# In[7]:


df.drop(columns=['Unnamed: 0'],inplace=True)



# In[9]:


X = df.drop(['personality'], axis=1)
y = df['personality']


# In[12]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X)


# In[13]:


rf = RandomForestClassifier()
rf.fit(X_sc, y)


pickle.dump(rf, open("../ml_model.pkl", "wb"))
pickle.dump(scaler, open("../scaler.pkl", "wb"))


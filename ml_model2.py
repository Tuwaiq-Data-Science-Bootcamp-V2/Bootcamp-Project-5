#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df = pd.read_csv('PersonalityTest-Copy1.csv')


# In[4]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[5]:


df['personality'] = df['personality'].map({'dependable': 0,
                                          'extraverted': 1,
                                          'lively': 2,
                                          'responsible': 3,
                                           'serious': 4})


# In[8]:


X = df.drop(['personality'], axis=1)
y = df['personality']


# In[9]:


scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X)


# In[10]:


def relabel_personality(x):
    if x == 0 or x == 3:
        return 1
    else:
        return 0

relabel_personality_v = np.vectorize(relabel_personality)

Y = relabel_personality_v(y)


# In[13]:


rf = RandomForestClassifier()
rf.fit(X_sc, Y)


# In[ ]:


pickle.dump(rf, open("ml_model2.pkl", "wb"))
pickle.dump(scaler, open("scaler2.pkl", "wb"))


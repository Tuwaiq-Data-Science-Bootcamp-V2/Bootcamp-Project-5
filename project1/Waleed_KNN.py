#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# ## Loading the dataset

# In[2]:


dataset = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# ## Define X by selecting only the age and EstimatedSalary, and y with purchased column

# In[5]:


X=dataset[['Age','EstimatedSalary']]
y=dataset['Purchased']


# In[6]:


X.head()


# In[7]:


y.head()


# ## Print count of each label in Purchased column

# In[8]:


dataset['Purchased'].value_counts()


# ## Print Correlation of each feature in the dataset

# In[9]:


dataset.corr()


# # First: Logistic Regression model

# ## Split the dataset into Training set and Test set with test_size = 0.25 and random_state = 0

# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[11]:


X_train.shape


# In[12]:


X_test.shape


# In[13]:


y_train.shape


# In[14]:


y_test.shape


# ## Train the model with random_state = 0

# In[15]:


lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
pred = lr.predict(X_test)


# ## Print the prediction results

# In[16]:


print(pred)


# ## Create dataframe with the Actual Purchased and Predict Purchased

# In[17]:


one_dataset = pd.DataFrame(X_test)
Two_dataset = pd.DataFrame(y_test)
Two_dataset['Predict_Purchased'] = pred
test_dataset= pd.concat([one_dataset,Two_dataset],axis=1)
test_dataset.rename(columns={'Purchased':'Actual_Purchased'}, inplace=True)
test_dataset


# In[18]:


(test_dataset['Actual_Purchased'] == test_dataset['Predict_Purchased']).value_counts()


# In[ ]:


pickle.dump(lr, open("LR_P.pkl", "wb"))


# ## Print Confusion Matrix and classification_report

# In[19]:

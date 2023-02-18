#!/usr/bin/env python
# coding: utf-8

# In[250]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


data = pd.read_csv('cars_raw.csv')
data


# In[55]:


data.shape


# In[35]:


data['DealType'].ffill(axis=0, inplace=True)


# In[26]:


data['Make'].value_counts()


# In[53]:


df = data.copy()
df.drop(columns=['ConsumerRating','ConsumerReviews','SellerRating','SellerType','SellerName','StreetName','State','Zipcode','ComfortRating','InteriorDesignRating','PerformanceRating','ValueForMoneyRating','ReliabilityRating','MinMPG','MaxMPG','Transmission','Engine','Stock#'],inplace=True)


# In[66]:


df.head()


# In[54]:


df.shape


# In[58]:


df.info()


# In[57]:


df.columns


# In[64]:


#df['FuelType']= df['FuelType'].replace(to_replace=['Individual','Dealer'],value=['Cash','Installments'])
df['FuelType'].value_counts()


# In[69]:


df['DealType']= df['DealType'].replace(to_replace=['Good','Great','Fair'],value=['Cash','Installments','Cash'])
df['DealType'].value_counts()


# In[70]:


df.rename(columns={'DealType':'BuyType'})


# In[85]:


df[['test','Fprice']] = df['Price'].str.split("$", expand = True)


# In[86]:


df


# In[87]:



df.drop(columns='test',inplace=True)


# In[88]:


df['Price'] = df['Fprice']


# In[102]:


df


# In[116]:


#df = df.astype({'Price':'float'})
df.drop(columns='Fprice')
df[['test','Fprice']] = df['Price'].str.split(",", expand = True)
df['prices'] = df['test'].astype(str)+df['Fprice']


# In[119]:


df.drop(columns=['Fprice','test','Price'],inplace=True)


# In[123]:


df.rename(columns={'prices':'Price'},inplace=True)


# In[131]:


df['Price']=df['Price'].astype(float)


# In[132]:


df.info()


# In[133]:


df.dtypes


# In[134]:


df['Price'].value_counts()


# In[136]:


df.head()


# In[135]:


df.columns


# In[138]:


df.drop(columns=['SellerReviews'],inplace=True)


# In[140]:


df['FuelType'].value_counts()


# In[143]:


df['FuelType'].replace(to_replace=['Gasoline', 'Gasoline Fuel', 'Electric Fuel System',
       'E85 Flex Fuel', 'Electric', 'Hybrid', 'Plug-In Electric/Gas', '–',
       'Flex Fuel Capability', 'Diesel', 'Diesel Fuel',
       'Gasoline/Mild Electric Hybrid', 'Flexible Fuel'],value=['Gasoline','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel','Diesel'],inplace=True)



# In[151]:


df.head()


# In[175]:


df.drop(columns=['Model','ExteriorColor','InteriorColor','Drivetrain','VIN','Mileage'],axis=1,inplace=True)


# In[176]:


df.head()


# In[ ]:


testList = ['Toyota', 'Ford', 'RAM', 'Honda', 'Lexus', 'Mercedes-Benz',
       'Dodge', 'Subaru', 'Acura', 'BMW', 'Audi', 'Volvo', 'Lincoln',
       'Land', 'Chevrolet', 'INFINITI', 'Tesla', 'Jeep', 'Chrysler',
       'Mazda', 'Kia', 'Volkswagen', 'Porsche', 'Nissan', 'Hyundai',
       'GMC', 'Buick', 'Genesis', 'Cadillac', 'Alfa', 'FIAT', 'Jaguar',
       'MINI', 'Lamborghini', 'Maserati', 'Mitsubishi', 'Bentley',
       'Mercury', 'Scion', 'Saturn', 'Ferrari']


# In[249]:


df['Make'].unique()


# In[190]:


df.drop(columns=['Used/New'],axis=1,inplace=True)


# In[191]:


df.dtypes


# In[192]:


from sklearn import preprocessing
df_corr =df.select_dtypes(include='object')
le = preprocessing.LabelEncoder()
df_corr = df_corr.apply(le.fit_transform)
df_corr.head()


# In[193]:


df1 = df.drop(df_corr.columns,axis=1)
newData = pd.concat([df_corr,df1],axis=1)
newData.head()


# In[203]:


np.isnan(newData.any())


# In[213]:


np.isfinite(newData.all())


# In[214]:


df_new = newData[np.isfinite(newData).all(1)]


# In[215]:


X = df_new.drop(columns=['Price'],axis=1)
y = df_new['Price']


# In[216]:


X.head()


# In[247]:


y.head()


# In[218]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[233]:


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
#dt = DecisionTreeRegressor()
sc = StandardScaler()
lr = LinearRegression()
x_train = sc.fit_transform(X_train)
x_test = sc.fit(X_test)
lr.fit(x_train,y_train)
y_pred = lr.predict(X_test)


# In[238]:


lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.intercept_)


# In[239]:


print(lr.intercept_)


# In[240]:


Coeff_df = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
Coeff_df


# In[241]:


preductions = lr.predict(X_test)
preductions


# In[242]:


test = pd.DataFrame(preductions)
test2 = pd.concat([test,y_test],axis=1)
test2


# In[ ]:


pickle.dump(lr, open("LR.pkl", "wb"))
pickle.dump(sc, open("SC.pkl", "wb"))


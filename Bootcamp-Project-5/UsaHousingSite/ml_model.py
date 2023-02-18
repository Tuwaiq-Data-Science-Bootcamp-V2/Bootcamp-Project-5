import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
lm = LinearRegression()
lm.fit(X,y)
pickle.dump(lm, open("ml_model.pkl", "wb"))
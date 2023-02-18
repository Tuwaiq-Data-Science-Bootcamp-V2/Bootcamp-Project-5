from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
# Create your views here.

def home(request):
    return render(request, 'index.html')

def getPredictions(Make, DealType, FuelType,Year, ExteriorStylingRating):
    lr = pickle.load(open('LR.pkl', 'rb'))
    sc = pickle.load(open('SC.pkl', 'rb'))
    le = preprocessing.LabelEncoder()
    #if DealType == 'Cash':
    #    DealType=0
    #else:
    #    DealType=1

    #if FuelType=='Gasoline':
    #    FuelType=1
    #else:
    #    FuelType=0
    dicts = {'Make':[Make],
             'DealType':[DealType],
             'FuelType':[FuelType],
             'Year': [Year],
             'ExteriorStylingRating':[ExteriorStylingRating]
             }
    print(dicts)
    df = pd.DataFrame(dicts,columns=['Make','DealType', 'FuelType', 'Year', 'ExteriorStylingRating'])
    print(df.dtypes)
    df_corr = df.select_dtypes(include='object')
    #print(df_corr)
    df_corr = df_corr.apply(le.fit_transform)
    #print(df_corr)
    prediction = lr.predict(sc.transform(df_corr))


    return prediction

def result(requset):
    Make = requset.GET['Make']
    Year = requset.GET['Year']
    DealType = requset.GET['DealType']
    FuelType = requset.GET['FuelType']
    ExteriorStylingRating = requset.GET['ExteriorStylingRating']

    result = getPredictions(Make,  DealType, FuelType,Year, ExteriorStylingRating)
    return render(requset,'result.html',{'result':result})
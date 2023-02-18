import numpy as np
from django.shortcuts import render
import pickle

# Create your views here.

def home(request):
    return render(request, 'index.html')

def getPredictions(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope,ca,thal):
    model = pickle.load(open('model.pkl', 'rb'))

    sex_en = pickle.load(open('sex.pkl','rb'))
    cp_en = pickle.load(open('cp.pkl', 'rb'))
    fbs_en = pickle.load(open('fbs.pkl', 'rb'))
    restecg_en = pickle.load(open('restecg.pkl', 'rb'))
    exang_en = pickle.load(open('exang.pkl', 'rb'))
    slope_en = pickle.load(open('slope.pkl', 'rb'))
    thal_en = pickle.load(open('thal.pkl', 'rb'))

    sex = sex_en.transform([sex])[0]
    fbs = fbs_en.transform([fbs])[0]
    cp = cp_en.transform([cp])[0]
    restecg = restecg_en.transform([restecg])[0]
    exxang = exang_en.transform([exang])[0]
    slope = slope_en.transform([slope])[0]
    thal = thal_en.transform([thal])[0]

    inp=[age,sex,cp,trestbps,chol,fbs,restecg,thalch,exxang,oldpeak,
         slope,ca,thal]

    prediction = model.predict([inp])[0]
    if prediction < 0.421:
        prediction=0
    else:
        prediction=1

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):
    age = int(request.GET['age'])
    sex = str(request.GET['sex'])
    cp = str(request.GET['cp'])
    trestbps = float(request.GET['trestbps'])
    chol = float(request.GET['chol'])
    fbs = str(request.GET['fbs'])
    if fbs=='True':
        fbs=True
    elif fbs=='False':
        fbs=False
    else:
        fbs=np.nan
    restecg = str(request.GET['restecg'])
    if restecg=='nan':
        restecg=np.nan
    thalch = float(request.GET['thalch'])
    exang = str(request.GET['exang'])
    if exang=='True':
        exang=True
    elif exang=='False':
        exang=False
    else:
        exang=np.nan
    oldpeak = float(request.GET['oldpeak'])
    slope = str(request.GET['slope'])
    if slope == 'nan':
        slope=np.nan
    ca = float(request.GET['ca'])
    thal = str(request.GET['thal'])
    if thal == 'nan':
        thal=np.nan

    result = getPredictions(age, sex, cp, trestbps,
                            chol, fbs,
                            restecg, thalch,
                            exang, oldpeak,
                            slope,ca,thal)

    return render(request, 'result.html', {'result': result})
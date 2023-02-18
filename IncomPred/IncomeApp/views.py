from django.shortcuts import render
import pickle
import numpy as np


def home(request):
    return render(request, 'index.html')


def getPredictions(blist):
    print(blist)
    blist=[74,10,0,3683,20,5,2,4,0]
    model = pickle.load(open('ml_model.pkl', 'rb'))
    # scaled = pickle.load(open('scaler.pkl', 'rb'))

    # prediction = model.predict(scaled.transform([
    #     [age, education, gain, loss,
    #    hours, workclass,
    #    relationship, race, sex]
    prediction = model.predict([np.array(blist)])
    if prediction[0] == 0:
        return 'no'

    elif prediction[0] == 1:
         return 'yes'

    else:
         return 'error'


def result(request):
    columns = '''age,education,gain,loss,hours,workclass,relationship,race,sex'''.split(',')
    fc=lambda x: int(request.GET[x])
    retlist = []
    # retlist = [55,99999,0,37,2,6,4,4,0]
    for c in columns:
        retlist.append(fc(c))


    result = getPredictions(retlist)

    return render(request, 'result.html', {'result': result})
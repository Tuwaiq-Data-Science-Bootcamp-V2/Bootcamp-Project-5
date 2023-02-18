from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')


def getPredictions(age, parch):
    model = pickle.load(open('/Users/shatha_95/Desktop/purchased_prediction/ml_model.pkl', 'rb'))
    scaled = pickle.load(open('/Users/shatha_95/Desktop/purchased_prediction/scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [age,parch]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):

    age = int(request.GET['age'])
    parch = int(request.GET['parch'])

    result = getPredictions(age,parch)

    return render(request, 'result.html', {'result': result})
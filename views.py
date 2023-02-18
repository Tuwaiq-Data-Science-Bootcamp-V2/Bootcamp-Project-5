from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(precipitation, temp_max, temp_min, wind):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [precipitation, temp_max, temp_min, wind]
    ]))

    if prediction == 0:
        return 'drizzle'
    elif prediction == 1:
        return 'fog'
    elif prediction == 2:
        return 'rain'
    elif prediction == 3:
        return 'snow'
    elif prediction == 4:
        return 'sun'
    else:
        return 'error'


def result(request):
    precipitation = float(request.GET['precipitation'])
    temp_max = float(request.GET['temp_max'])
    temp_min = float(request.GET['temp_min'])
    wind = float(request.GET['wind'])

    result = getPredictions(precipitation, temp_max, temp_min, wind)

    return render(request, 'result.html', {'result': result})
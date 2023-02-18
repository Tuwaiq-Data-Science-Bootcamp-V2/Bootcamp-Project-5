
from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion):
    model = pickle.load(open('ml_model2.pkl', 'rb'))
    scaled = pickle.load(open('scaler2.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'


def result(request):

    gender = int(request.GET['gender'])
    age = int(request.GET['age'])
    openness = int(request.GET['openness'])
    neuroticism = int(request.GET['neuroticism'])
    conscientiousness = int(request.GET['conscientiousness'])
    agreeableness = int(request.GET['agreeableness'])
    extraversion = int(request.GET['extraversion'])



    result = getPredictions(gender, age, openness, neuroticism,
                            conscientiousness, agreeableness, extraversion)

    return render(request, 'result.html', {'result': result})

# Create your views here.

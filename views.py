from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]
    ]))

    if prediction == 'extraverted':
        return 'extraverted'
    elif prediction == 'serious':
        return 'serious'
    elif prediction == 'dependable':
        return 'dependable'
    elif prediction == 'lively':
        return 'lively'
    elif prediction == 'responsible':
        return 'responsible'
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
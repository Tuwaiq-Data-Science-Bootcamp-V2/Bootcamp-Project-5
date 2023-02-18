from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(gender,age,salary):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [gender,age,salary]
    ]))
    # prediction = model.predict([[gender,age,salary]])

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'


def result(request):
    gender = int(request.GET['gender'])
    age = int(request.GET['age'])
    salary = int(request.GET['salary'])

    result = getPredictions(gender,age,salary)

    return render(request, 'result.html', {'result': result})
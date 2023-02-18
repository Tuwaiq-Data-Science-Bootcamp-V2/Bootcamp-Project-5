from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'base.html')

def getPredictions(age,salary):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([[age, salary]]))

    if prediction == 0:
        return 'Not Purchased'
    elif prediction == 1:
        return 'Purchased'
    else:
        return 'Error'

def result(request):
    age = int(request.GET['age'])
    salary = int(request.GET['salary'])

    result = getPredictions(age, salary)

    return render(request, 'result.html', {'result': result})





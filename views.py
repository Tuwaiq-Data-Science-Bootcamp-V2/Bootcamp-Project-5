from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(gender,degree_percentage,work_experience,emp_test_percentage):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [gender, degree_percentage, work_experience, emp_test_percentage]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'


def result(request):
    gender = int(request.GET['gender'])
    degree_percentage = int(request.GET['degree_percentage'])
    work_experience = int(request.GET['work_experience'])
    emp_test_percentage = int(request.GET['emp_test_percentage'])


    result = getPredictions(
        gender,degree_percentage,work_experience,emp_test_percentage)

    return render(request, 'result.html', {'result': result})
from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'beasd.html')



def getPredictions(Gender,Age,EstimatedSalary):
    model = pickle.load(open('lm.pkl', 'rb'))
    scaled = pickle.load(open('lr.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [Gender,Age,EstimatedSalary]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'


def result(request):
    Gender= int(request.GET['Gender'])
    Age = int(request.GET['Age'])
    EstimatedSalary = int(request.GET['EstimatedSalary'])


    result = getPredictions(Gender,Age,EstimatedSalary)

    return render(request, 'result.html', {'result': result})

# Create your views here.

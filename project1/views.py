from django.shortcuts import render
import pickle
# Create your views here.
def home(request):
    return render(request, 'index.html')

def Model(Age,EstimatedSalary):
    lr = pickle.load(open('LR_P.pkl', 'rb'))
    pred = lr.predict([[Age,EstimatedSalary]])

    if pred ==0:
        return 'Not Purchased'
    else:
        return 'Purchased'

def result(request):
    Age = int(request.GET['Age'])
    EstimatedSalary = int(request.GET['EstimatedSalary'])

    result = Model(Age,EstimatedSalary)
    return render(request, 'result.html', {'result': result})
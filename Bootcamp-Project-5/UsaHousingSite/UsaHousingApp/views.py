from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getpredictions(avg_income, avg_age, avg_room, avg_bedroom, pop):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    prediction = model.predict([
        [avg_income, avg_age, avg_room, avg_bedroom, pop]
    ])
    return prediction


def result(request):
    avg_income = float(request.GET['avg_income'])
    avg_age = float(request.GET['avg_age'])
    avg_room = float(request.GET['avg_room'])
    avg_bedroom = float(request.GET['avg_bedroom'])
    pop = float(request.GET['pop'])
    results = getpredictions(avg_income, avg_age, avg_room, avg_bedroom, pop)
    return render(request, 'result.html', {'result': results})

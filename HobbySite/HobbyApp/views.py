from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(Olympiad_Participation, Scholarship, School, Projects, Grasp_pow, Time_sprt, Medals,
                   Career_sprt, Act_sprt, Fant_arts, Time_art):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    #encoder = pickle.load(open('labelEnc.pkl', 'rb'))

    prediction = model.predict([
        [Olympiad_Participation, Scholarship, School, Projects, Grasp_pow, Time_sprt, Medals, Career_sprt,
         Act_sprt, Fant_arts, Time_art]
    ])

    if prediction == 0:
        return 'Academics'
    elif prediction == 1:
        return 'Sports'
    else:
        return 'Arts'


def result(request):
    Olympiad_Participation = int(request.GET['Olympiad_Participation'])
    Scholarship = int(request.GET['Scholarship'])
    School = int(request.GET['School'])
    Projects = int(request.GET['Projects'])
    Grasp_pow = int(request.GET['Grasp_pow'])
    Time_sprt = int(request.GET['Time_sprt'])
    Medals = int(request.GET['Medals'])
    Career_sprt = int(request.GET['Career_sprt'])
    Act_sprt = int(request.GET['Act_sprt'])
    Fant_arts = int(request.GET['Fant_arts'])
    Time_art = int(request.GET['Time_art'])

    result = getPredictions(Olympiad_Participation, Scholarship, School, Projects, Grasp_pow,
                            Time_sprt, Medals, Career_sprt, Act_sprt, Fant_arts, Time_art)

    return render(request, 'result.html', {'result': result})

# Create your views here.

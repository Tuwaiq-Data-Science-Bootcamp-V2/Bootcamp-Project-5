from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')
# Create your views here.

def getPredictions(creditScore, county, gender, age,tenure, balnace
                            ,numOfProducts, activeMember,credictCard,estSalary):

    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))
    if gender == 'Female' or gender =='female':
        new_gender = 0
    elif gender=='Male' or gender =='male':
        new_gender = 1

    if county == 'France' or county =='france':
        new_country = 0
    elif county=='Germany' or county =='germany':
        new_country = 1
    elif county=='Spain' or county =='spain':
        new_country = 2

    if activeMember == 'Yes' or activeMember == 'yes':
        new_activeMember = 1
    elif activeMember == 'No' or activeMember == 'no':
        new_activeMember = 0

    if credictCard == 'Yes' or credictCard == 'yes':
        new_credictCard = 1
    elif credictCard == 'No' or credictCard == 'no':
        new_credictCard = 0

    prediction = model.predict(scaled.transform([[creditScore, new_country, new_gender, age,tenure, balnace
                            ,numOfProducts, new_activeMember,new_credictCard,estSalary]]))

    if prediction == 1:
        return 'no'
    elif prediction == 0:
        return 'yes'
    else:
        return 'error'


def result(request):
    creditScore = int(request.GET['creditScore'])
    county = request.GET['county']
    gender = request.GET['gender']
    age = int(request.GET['age'])
    tenure = int(request.GET['tenure'])
    balnace = float(request.GET['balnace'])
    numOfProducts = int(request.GET['numOfProducts'])
    activeMember = request.GET['activeMember']
    credictCard = request.GET['credictCard']
    estSalary = float(request.GET['estSalary'])



    result = getPredictions(creditScore, county, gender, age,tenure, balnace
                            ,numOfProducts, activeMember,credictCard,estSalary)

    return render(request, 'result.html', {'result': result})
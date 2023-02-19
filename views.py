from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'


def result(request):
    Gender = int(request.GET['Gender'])
    Age = int(request.GET['Age'])
    Driving_License = int(request.GET['Driving_License'])
    Region_Code = int(request.GET['Region_Code'])
    Previously_Insured = int(request.GET['Previously_Insured'])
    Vehicle_Age = int(request.GET['Vehicle_Age'])
    Vehicle_Damage = int(request.GET['Vehicle_Damage'])
    Annual_Premium = int(request.GET['Annual_Premium'])
    Policy_Sales_Channel = int(request.GET['Policy_Sales_Channel'])
    Vintage = int(request.GET['Vintage'])


    result = getPredictions(Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage)

    return render(request, 'result.html', {'result': result})
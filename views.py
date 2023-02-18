from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPrediction(gender,ssc_percentage,hsc_percentage,hsc_subject,degree_percentage, undergrad_degree, work_experience,emp_test_percentage, specialisation,mba_percent):

    model = pickle.load(open('Faris2.pkl','rb'))

    prediction = model.predict([
        [gender,ssc_percentage,hsc_percentage,hsc_subject,degree_percentage, undergrad_degree, work_experience,emp_test_percentage, specialisation,mba_percent]
    ])

    if prediction == 'Not Placed':
        return 'no'
    elif prediction == 'Placed':
        return 'yes'
    else:
        return 'error'


def result(request):
    gender = int(request.GET['gender'])
    hsc_subject = int(request.GET['hsc_subject'])
    undergrad_degree = int(request.GET['undergrad_degree'])
    work_experience = int(request.GET['work_experience'])
    specialisation = int(request.GET['specialisation'])
    ssc_percentage = float(request.GET['ssc_percentage'])
    hsc_percentage = float(request.GET['hsc_percentage'])
    degree_percentage = float(request.GET['degree_percentage'])
    emp_test_percentage = float(request.GET['emp_test_percentage'])
    mba_percent = float(request.GET['mba_percent'])
    results = getPrediction(gender,ssc_percentage,hsc_percentage,hsc_subject,degree_percentage, undergrad_degree, work_experience,emp_test_percentage, specialisation,mba_percent)

    return render(request, 'result.html', {'result': results})
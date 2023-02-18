from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def home(request):
    return render(request, 'index.html')

def getPredictions(experience_level,employment_type,job_title,
                    employee_residence,remote_ratio,company_location,company_size):
    model = pickle.load(open('model.pkl', 'rb'))
    
    
    experience_level_en = pickle.load(open('experience_level.pkl','rb'))
    employment_type_en = pickle.load(open('employment_type.pkl','rb'))
    job_title_en = pickle.load(open('job_title.pkl','rb'))
    employee_residence_en = pickle.load(open('employee_residence.pkl','rb')) 
    company_location_en = pickle.load(open('company_location.pkl','rb')) 
    company_size_en = pickle.load(open('company_size.pkl','rb')) 
    
    experience_level = experience_level_en.transform([experience_level])[0]
    employment_type = employment_type_en.transform([employment_type])[0]
    job_title = job_title_en.transform([job_title])[0]
    employee_residence = employee_residence_en.transform([employee_residence])[0]
    company_location = company_location_en.transform([company_location])[0]
    company_size = company_size_en.transform([company_size])[0]
    
    
    prediction = model.predict(([
                     [experience_level,employment_type,job_title,
                    employee_residence,remote_ratio,company_location,company_size]
                    ]))[0]
                
    if prediction == 1:
        return 'yes'
    elif prediction == 0:
        return 'no'
    else:
        return 'error'
    
def result(request):

    experience_level = (request.GET['experience_level'])
    employment_type = (request.GET['employment_type'])
    job_title = (request.GET['job_title'])
    employee_residence = (request.GET['employee_residence'])
    remote_ratio = int(request.GET['remote_ratio'])
    company_location = (request.GET['company_location'])
    company_size = (request.GET['company_size'])
    
    result = getPredictions(experience_level,employment_type,job_title,
                employee_residence,remote_ratio,company_location,company_size)

    return render(request, 'result.html', {'result': result})
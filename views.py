from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(slen, swidth, plen, pwidth):
    model = pickle.load(open('tree_model.pkl', 'rb'))
    scaled = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model.predict(scaled.transform([
        [slen, swidth, plen, pwidth]
    ]))

    if prediction == 0:
        return 'Setosa'
    elif prediction == 1:
        return 'Versicolor'
    elif prediction == 2:
        return 'Virginica'
    else:
        return 'error'


def result(request):
    slen = float(request.GET['slen'])
    swidth = float(request.GET['swidth'])
    plen = float(request.GET['plen'])
    pwidth = float(request.GET['pwidth'])

    result = getPredictions(slen, swidth, plen, pwidth)

    return render(request, 'result.html', {'result': result})
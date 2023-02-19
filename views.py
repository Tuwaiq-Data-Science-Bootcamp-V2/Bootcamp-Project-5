from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')


def getPredictions(op, scoler, school, fav_sub, porj, Grasp_p, tsprt,medals,csprt,
    actsprt,farts,warts,tarts):
    lr = pickle.load(open('lr.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    lra = pickle.load(open('lra.pkl', 'rb')) 
    lrs = pickle.load(open('lrs.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    sportpred = lrs.predict([[op,scoler,school,fav_sub,porj,
    Grasp_p,tsprt,medals,csprt,
    actsprt,farts,warts,tarts]])
    
    artpred = lra.predict([[op,scoler,school,fav_sub,porj,
    Grasp_p,tsprt,medals,csprt,
    actsprt,farts,warts,tarts]])
    
    acdpred = lr.predict([[op,scoler,school,fav_sub,porj,
    Grasp_p,tsprt,medals,csprt,
    actsprt,farts,warts,tarts]])

    lastpred = model.predict([[op,scoler,school,fav_sub,porj,
    Grasp_p,tsprt,medals,csprt,
    actsprt,farts,warts,tarts]])
    

    if sportpred == 1:
        return 'Sports'
    elif acdpred == 1:
        return 'Academics'
    elif artpred == 1:
        return 'Arts'
        
    else:
        return lastpred[0]


def result(request):
    op = int(request.GET['op'])
    scoler = int(request.GET['scoler'])
    school = int(request.GET['school'])
    fav_sub = int(request.GET['fav_sub'])
    porj = int(request.GET['porj'])
    Grasp_p = int(request.GET['Grasp_p'])
    tsprt = int(request.GET['tsprt'])
    medals = int(request.GET['medals'])
    csprt = int(request.GET['csprt'])
    actsprt = int(request.GET['actsprt'])
    farts = int(request.GET['farts'])
    warts = int(request.GET['warts'])
    tarts = int(request.GET['tarts'])
    

    result = getPredictions(op, scoler, school, fav_sub, porj, Grasp_p, tsprt,medals,csprt,
    actsprt,farts,warts,tarts)

    return render(request, 'result.html', {'result': result})
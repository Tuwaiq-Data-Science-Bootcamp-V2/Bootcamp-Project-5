from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.html')

def getPredictions(num_posts,num_following,num_followers,biography_length,picture_availability,link_availability,average_caption_length):
    model = pickle.load(open('ml_model.pkl', 'rb'))
    prediction = model.predict([[num_posts,num_following,num_followers,biography_length,picture_availability,link_availability,average_caption_length]])

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):
    num_posts = int(request.GET['num_posts'])
    num_following = int(request.GET['num_following'])
    num_followers = int(request.GET['num_followers'])
    biography_length = int(request.GET['biography_length'])
    picture_availability = request.GET['picture_availability']
    if picture_availability=='yes':
        picture_availability=1
    else:
        picture_availability=0
    link_availability = request.GET['link_availability']
    if link_availability == 'yes':
        link_availability = 1
    else:
        link_availability = 0

    average_caption_length = int(request.GET['average_caption_length'])


    result = getPredictions(num_posts,num_following,num_followers,biography_length,picture_availability,link_availability,average_caption_length)

    return render(request, 'result.html', {'result': result})
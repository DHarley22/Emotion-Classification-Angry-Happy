from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth
import torch
import numpy as np
from keras.models import load_model
from face.models import Photo, FaceDetection
from django.urls import reverse
from . import forms
import cv2
from PIL import Image
from torchvision import transforms
from face.model import EmotionClassifier
import os

# Charger le modèle d'émotion Keras
emotion_model = load_model('/home/harley/Documents/AHFace-master/face/best_model.keras')
# Charger le modèle d'émotion PyTorch
model_path = os.path.join(os.path.dirname(__file__), 'best_model_efficient.pth')
classifier = EmotionClassifier(model_path)

# Définir le dispositif (CPU ou GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_image(image, model, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image).item()
        prediction = 'Happy' if output > threshold else 'Angry'
        return prediction

def home(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        photo = Photo.objects.create(face = image_file)
        # np_image = np.frombuffer(image_file.read(), np.uint8)
        np_image = np.frombuffer(photo.face.read(), np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
        # Détection des visages
        face_detection_model = cv2.CascadeClassifier('/home/harley/Documents/AHFace-master/face/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detection_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_image = image[y:y+h, x:x+w]
                face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))  # Convertir en image PIL

                prediction = predict_image(face_image, classifier.model)
                print(f"Emotion detected: {prediction}")

                context = {
                    'prediction': prediction,
                    'image': photo.face.url
                
                }
                return render(request, 'face/resultat.html', context)
        else:
            message = 'Cette image ne contient pas de visage'
            return render(request, 'face/acceuil.html', {'message': message})
    
    return render(request, 'face/acceuil.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username is already taken')
                return redirect(register)
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email is already taken')
                return redirect(register)
            else:
                user = User.objects.create_user(username=username, password=password, email=email, first_name=first_name, last_name=last_name)
                user.save()
                return redirect('login_user')
        else:
            messages.info(request, 'Both passwords are not matching')
            return redirect(register)
    else:
        return render(request, 'face/registration.html')

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('home')
        else:
            messages.info(request, 'Invalid Username or Password')
            return redirect('login_user')
    else:
        return render(request, 'face/login.html')

def logout_user(request):
    auth.logout(request)
    return redirect('home')

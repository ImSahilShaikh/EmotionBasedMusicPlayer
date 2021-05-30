from django.shortcuts import render

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
from django.shortcuts import redirect

# Create your views here.

val = 1

def home(request):
    return render(request, "index.html")


def runApp(request):
    global val
    mypath = os.path.join(os.getcwd(),'EMP','input_data')

    def CaptureImages():
        global val
        cam = cv2.VideoCapture(0)
        ret,frame = cam.read()
        cv2.imwrite(os.path.join(mypath,'cam_images/')+str(val)+'.jpg',frame)
        val = val + 1
        cam.release()

    CaptureImages()          
    face_classifier = cv2.CascadeClassifier(os.path.join(mypath,'face_detector','haarcascade_frontalface_default.xml'))
    print("face detector loaded")
    classifier = load_model(os.path.join(os.getcwd(),'EMP','model.h5'))

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # cap = cv2.VideoCapture(0)

    getimage = os.listdir(os.path.join(mypath,'cam_images/'))
    print(getimage)
    #image = cv2.imread(os.path.join(os.path.join(mypath,'cam_images/'),getimage[0]))
    image = cv2.imread(os.path.join(os.path.join(mypath,'images/'),"happy1.jpg"))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = []
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    print(faces)

    if( len(faces)==0):
        return render(request,"error.html")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

    preds = classifier.predict(roi)[0]
    print("\nprediction = ", preds)
    label = class_labels[preds.argmax()]
    print("\nprediction max = ", preds.argmax())
    print("\nlabel = ", label)
    label_position = (x+20, y-20)
    cv2.putText(image, label, label_position,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #cv2.imshow("Emotion Detector", image)
    #cv2.waitKey(0)

    #CHECK THIS
    # for f in os.path.join(mypath,'cam_images/'):
    #     os.remove(os.path.join(mypath,'cam_images/',f))

    return render(request,"loading.html",{'label':label})



$pip install numpy, pandas 
import numpy as np
import pandas as pd
#import os
#import time
#import matplotlib.pyplot as plt
import cv2
#import seaborn as sns
#sns.set_style('darkgrid')
#import shutil
#from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.layers import MaxPool2D
#from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization,Flatten
#from tensorflow.keras.optimizers import Adam,Adamax,RMSprop,SGD
#from tensorflow.keras.metrics import categorical_crossentropy
#from tensorflow.keras import regularizers
#from tensorflow.keras.models import Model
#from tensorflow.keras import models, layers, regularizers
#from tensorflow.keras.models import Sequential ,Model
#from tensorflow.keras.utils import to_categorical, plot_model
#from tensorflow.keras import layers
import os
from tensorflow.keras.models import load_model
#import csv
from PIL import Image

print("working dir:"+os.getcwd())
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

music_df =pd.read_csv("input/data_moods.csv")
model = load_model('model_optimal.h5')

Emotion_Classes = ['Angry', 
                  'Disgust', 
                  'Fear', 
                  'Happy', 
                  'Neutral', 
                  'Sad', 
                  'Surprise']

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  

# Making Songs Recommendations Based on Predicted Class
def Recommend_Songs(pred_class):
    
    if( pred_class=='Disgust' ):

        Play = music_df[music_df['mood'] =='Sad' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        #Play = Play[:5].reset_index(drop=True)
        #display(Play)

    if( pred_class=='Happy' or pred_class=='Sad' ):

        Play = music_df[music_df['mood'] =='Happy' ]
        Play = Play.sort_values(by="popularity", ascending=False)
       # Play = Play[:5].reset_index(drop=True)
        #display(Play)

    if( pred_class=='Fear' or pred_class=='Angry' ):

        Play = music_df[music_df['mood'] =='Calm' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        #Play = Play[:5].reset_index(drop=True)
        #display(Play)
        
    if( pred_class=='Surprise' or pred_class=='Neutral' ):

        Play = music_df[music_df['mood'] =='Energetic' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        #Play = Play[:5].reset_index(drop=True)
        #display(Play)
    print(Play)
# Write data to CSV file
    Play.to_csv('output/music.cvs', index=False)

def load_and_prep_image(filename, img_shape = 48):
    img = cv2.imread(filename)
    GrayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)
    for x,y,w,h in faces:
        roi_GrayImg = GrayImg[ y: y + h , x: x + w ]
        roi_Img = img[ y: y + h , x: x + w ]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        faces = faceCascade.detectMultiScale(roi_Img, 1.1, 4)
        if len(faces) == 0:
            print("No Faces Detected")
        else:
            for (ex, ey, ew, eh) in faces:
                img = roi_Img[ ey: ey+eh , ex: ex+ew ]
    Img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Img= cv2.resize(Img,(img_shape,img_shape))
    Img = Img/255.
    return Img

def pred_and_plot(filename, class_names):
    # Import the target image 
    img = load_and_prep_image(filename)
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0) , verbose=0)
    # Get the predicted class
    pred_class = class_names[pred.argmax()]
    print(pred_class)
    # Open the file in write mode
    with open('output/mood.txt', 'w') as file:
        file.write(pred_class)
    scaled_image_array = (img * 255).astype(np.uint8)
    image = Image.fromarray(scaled_image_array)
    image.save("output/mood.jpg")
    Recommend_Songs(pred_class)

pred_and_plot("input/mood.jpg",Emotion_Classes)


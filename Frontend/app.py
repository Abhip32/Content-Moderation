from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from keras.models import load_model
import keras_ocr 
pipeline = keras_ocr.pipeline.Pipeline()
import pandas as pd 
import numpy as np 
from glob import glob
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt 
from PIL import Image
import keras
import pickle
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,SpatialDropout1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.models import Sequential


IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
MoBiLSTM_model=load_model('../imageContent/model.h5')
load_model=keras.models.load_model("../profanity/hate&abusive_model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)
stopword=set(stopwords.words('english'))

CLASSES_LIST = ["NonViolence","Violence"]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

class UploadForm(FlaskForm):
    image = FileField('image')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        # Save the uploaded file
        image_file = request.files['image']
        im = Image.open(image_file)
        rgb_im = im.convert("RGB")
        rgb_im.save('./Images/'+image_file.filename)
        voilence_result=predict_video('./Images/'+image_file.filename, 16)
        results = pipeline.recognize(['./Images/'+image_file.filename])
        textInImage=[]
        for text, box in results[0]:
            textInImage.append(text)
        text=listToString(textInImage)
        test=[clean_text(text)]
        seq = load_tokenizer.texts_to_sequences(test)
        padded = pad_sequences(seq, maxlen=300)
        pred = load_model.predict(padded)
        if pred<0.5:
            hate_result="No Hate";
        else:
            hate_result="Hate and abusive"

        return render_template('result.html', hate_result=hate_result,voilence_result=voilence_result["predicted_class_name"],confidence=voilence_result["confidence"])
        
    return render_template('upload.html', form=form)



def predict_video(video_file_path, SEQUENCE_LENGTH):
 
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Store the predicted class in the video.
    predicted_class_name = ''
 
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
 
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        success, frame = video_reader.read() 
 
        if not success:
            break
 
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
    
    if ".jpg" in video_file_path:
        print(video_file_path)
        img = cv2.imread(video_file_path)

        frames_list.clear();

        for i in range(16):
            resized_frame = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)
        
 

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = MoBiLSTM_model.predict(np.expand_dims(frames_list, axis = 0))[0]
 
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
 
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]


    
    
    # Display the predicted class along with the prediction confidence.
    print(f'Predicted: {predicted_class_name} \nConfidence: {predicted_labels_probabilities[predicted_label]}')
    return {"predicted_class_name": predicted_class_name,"confidence": predicted_labels_probabilities[predicted_label]}
    



def clean_text(text):
    print(text)
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    print(text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


def listToString(s):
    str1 = ""
    for ele in s:
        str1= str1+" "+ele
    return str1
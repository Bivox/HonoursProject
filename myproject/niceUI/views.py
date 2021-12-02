from django.shortcuts import render

from myapp.models import Destination

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import base64
from flask import Flask, render_template, request, jsonify

# Create your views here.
def index(request):

   dest1 = Destination()
   dest1.price = 400

   dest2 = Destination()
   dest2.price = 620

   dests = [dest1, dest2]
   return render(request, "index.html", {'dests': dests})

# Initialize flask app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('app/mnist_classification.h5')

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('drawing.html')

# Handle POST request
@app.route('/', methods=['POST'])
def canvas():
    # Recieve base64 data from the user form
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (28, 28)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)

    # Expand to numpy array dimenstion to (1, 28, 28)
    img = np.expand_dims(gray_image, axis=0)

    try:
        prediction = np.argmax(model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render_template('drawing.html', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template('drawing.html', response=str(e), canvasdata=canvasdata)

def digit_rec_model(request):
   #hand written characters 28x28 sized images of 0...9
   mnist=tf.keras.datasets.mnist

   (x_train,y_train), (x_test,y_test)=mnist.load_data()

   #normalize
   x_train=tf.keras.utils.normalize(x_train,axis=1)
   x_test=tf.keras.utils.normalize(x_test,axis=1)

   # Resizing the image for Convolutional Operation
   IMG_SIZE=28
   #increasing dimension by 1 for kernel=filter operation
   x_trainr=np.array(x_train).reshape(-1, IMG_SIZE,IMG_SIZE,1)  #reshape(60000,28,28,1)
   x_testr=np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE,1)
   #print("Training Samples dimension",x_trainr.shape)
   #print("Testing Samples dimension",x_testr.shape)

   # Creating a DNN (Deep Neural Network)
   # Training with 60,000 samples
   # create a neural net
   model=Sequential()

   # first convolution layer  (60000,28,28,1) 28-3+1=26x26
   model.add(Conv2D(64,(3,3), input_shape=x_trainr.shape[1:])) # 64 filters with size of 3x3
   model.add(Activation("relu")) # activation function to make it non-linear, <0, remove, >0
   model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling single maximum value of 2x2

   # 2nd convolution layer    26-3+1=24x24
   model.add(Conv2D(64,(3,3)))
   model.add(Activation("relu"))
   model.add(MaxPooling2D(pool_size=(2,2)))

   # 3rd convolution layer
   model.add(Conv2D(64,(3,3)))  #13x13
   model.add(Activation("relu"))
   model.add(MaxPooling2D(pool_size=(2,2)))

   # Fully connected layer #1   20x20=400
   model.add(Flatten()) #before using fully connected layer, need to be flatten so that 2D to 1D
   model.add(Dense(64))    #each 400 will be connected to each 64 neurons
   model.add(Activation("relu"))

   # Fully connected layer #2   20x20=400
   model.add(Dense(32))    # decreasing the size gradually, we're trying to reach 10 bc we have labelled 10 digits 
   model.add(Activation("relu"))

   # Fully connected layer #3 (LAST)   20x20=400
   model.add(Dense(10))    # the last dense layer must be equal to 10
   model.add(Activation("softmax")) #activation with Softmax (can also be sigmoid for BINARY classification)(class probabilities, not for binary)
   # softmax is useful for probability distributions

   model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

   # Training the model
   model.fit(x_trainr, y_train, epochs=5,validation_split=0.3)

   # Predictions
   # preditions are an array of class probabilities, so we need to decode them
   predictions=model.predict([x_testr])

   # Save model
   model.save('digit_rec.h5')

   # Evaluating the predictions
   # Comparing test data vs predicted data

   return render(request, "index.html")
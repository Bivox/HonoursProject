from django.shortcuts import render
from tensorflow import keras

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
import cv2
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image

from itertools import product
import os
from django.shortcuts import render


# Create your views here.



## Initialize flask app
app = Flask(__name__)

# Load prebuilt model
#reconstructed_model = tf.lite.TFLiteConverter.from_keras_model('app/digit_rec.h5')
reconstructed_model = tf.keras.models.load_model('niceUI/digit_rec.h5')


# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('index.html')

# Handle POST request
@app.route('/', methods=['POST'])
def canvas(request):
    # Recieve base64 data from the user form
    #canvasdata = request.form["canvasimg"]
    #encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    #nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.imread('/Users/Marco/Downloads/hand_written_digit.png')

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (28, 28)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)

    # Expand to numpy array dimenstion to (1, 28, 28)
    img = np.expand_dims(gray_image, axis=0)

    try:
        prediction = np.argmax(reconstructed_model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render('index.html', {'prediction':prediction})
    except Exception as e:

        return render('index.html', {'prediction':prediction})

def load_model(request):
    # Load prebuilt model
    reconstructed_model = tf.keras.models.load_model('niceUI/digit_rec.h5')
    return render(request, 'index.html',{'new_model':reconstructed_model.get_weights})

def digit_rec_model():

    print("We've started...")
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

    model=Sequential()

    # first convolution layer  (60000,28,28,1) 28-3+1=26x26
    model.add(Conv2D(64,(3,3), input_shape=(28, 28, 1))) # 64 filters with size of 3x3
    model.add(LeakyReLU()) # activation function to make it non-linear, <0, remove, >0
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling single maximum value of 2x2

    # 2nd convolution layer    26-3+1=24x24
    model.add(Conv2D(64,(3,3)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 3rd convolution layer
    model.add(Conv2D(64,(3,3)))  #13x13
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Fully connected layer #1   20x20=400
    model.add(Flatten()) #before using fully connected layer, need to be flatten so that 2D to 1D


    # Fully connected layer #2   20x20=400 
  

    # Fully connected layer #3 (LAST)   20x20=400
    model.add(Dense(10))    # the last dense layer must be equal to 10
    model.add(Activation("softmax")) #activation with Softmax (can also be sigmoid for BINARY classification)(class probabilities, not for binary)
    # softmax is useful for probability distributions

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

    # Training the model
    model.fit(x_trainr, y_train, epochs=5,validation_split=0.1)

    # Predictions
    # preditions are an array of class probabilities, so we need to decode them
    predictions=model.predict([x_testr])

    
    # Save model
    model.save('niceUI/digit_rec.h5')

    # Creating a DNN (Deep Neural Network)
    # Training with 60,000 samples
    # create a neural net
    '''model=Sequential()

    # first convolution layer  (60000,28,28,1) 28-3+1=26x26
    model.add(Conv2D(64,(3,3), input_shape=(28, 28, 1))) # 64 filters with size of 3x3
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
    model.fit(x_trainr, y_train, epochs=5,validation_split=0.1)

    # Predictions
    # preditions are an array of class probabilities, so we need to decode them
    predictions=model.predict([x_testr])
    
    # Save model
    model.save('niceUI/digit_rec.h5')'''

    # Evaluating the predictions
    # Comparing test data vs predicted data

    return model

def crop(img, x = 0, y = 0, w = 800, h = 800):

    # Setting the points for cropped image
    #for first section

    im1 = img[y:h+y, x:w+x]
    print()
    #print("img shape:",im1.shape)
    '''#cropping second half of image
    x1=w//2
    im2 = img[y:h, x1:w]'''
    return im1


def predict_digit(request):
    # Load prebuilt model
    reconstructed_model = tf.keras.models.load_model('niceUI/digit_rec.h5')
    #reconstructed_model = digit_rec_model()
    IMG_SIZE=28
    img = cv2.imread('/Users/Marco/Downloads/hand_written_digit.png')
    #print("img shape:",img.shape)
    #TODO img_left, img_right = crop(img)
    imgray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgray_1, 127, 255, 0)
    thresh = cv2.Canny(imgray_1, 100, 100 * 2)
    #canny_output=cv2.blur(canny_output, (5,5))
    contours = []
    contours_tmp = cv2.findContours(imgray_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_tmp = contours_tmp[0] if len(contours_tmp) == 2 else contours_tmp[1]
    imgray_1_copy = imgray_1.copy()
    while 0 != len(contours_tmp):
        contours.append(contours_tmp[0])
        x, y, w, h = cv2.boundingRect(contours_tmp[0])
        imgray_1_copy = cv2.rectangle(imgray_1_copy, (x,y), (x+w,y+h), (0,0,0), -1)
        contours_tmp = cv2.findContours(imgray_1_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_tmp = contours_tmp[0] if len(contours_tmp) == 2 else contours_tmp[1]
        pass
    #creating a list of cropped imgs
    lst = []
    coord_lst = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        coord_lst.append(x)
        #print(x, y, w, h)
        canvas_width = w + 50 if w + 50 > 300 else 300
        canvas_height = h + 50
        x_center = (canvas_width - w) // 2
        y_center = (canvas_height - h) // 2
        cropped = crop(imgray_1, x, y, w, h)
        print(x, y)
        print("--------- cropped dtype",cropped.shape)

        result_image = np.full((canvas_height,canvas_width,1),0, dtype=np.uint8)
        print("1----result_image shape:",result_image.shape)
        # copy img image into center of result image
        cropped = np.expand_dims(cropped, axis=-1)
        result_image[y_center:y_center+h, x_center:x_center+w] = cropped
        #result_image[0:0 + h, 0:0+w] = cropped

        cv2.imshow("dying inside", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        lst.append(result_image)
    #cv2.imshow("canny", canny_output)
    print("length of con: "+str(len(contours)))
    '''if cv2.countNonZero(img_right) == 0:
        print ("Image is black")
    else:
        print ("Colored image")'''
    result=""

    res = dict(zip(coord_lst, lst))
    lst=dict(sorted(res.items(),key= lambda x:x[0])).values()
    for c in lst:
        
        resized=cv2.resize(c, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imshow("dying inside", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 0 to 1 scaling
        norm_img=tf.keras.utils.normalize(resized,axis=1) 
        #kernel operation of convolution layer
        norm_img=np.array(norm_img).reshape(-1, IMG_SIZE, IMG_SIZE,1) 
        predictions=reconstructed_model.predict(norm_img)
        result += str(np.argmax(predictions))
        print("predicted value: "+str(np.argmax(predictions)))
    if result == "":
        print("we facked up m8")
    return render(request, "index.html", {'prediction_number':result,'model':reconstructed_model.get_weights})



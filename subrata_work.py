# -*- coding: utf-8 -*-
"""Subrata_work.ipynb


"""

import time
import math
from math import ceil
import numpy as np
import os,cv2

import tensorflow as tf
import keras


import numpy as np
np.random.seed(1337)
#import matplotlib.pyplot as plt

#from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from PIL import Image
#from keras.utils import np_utils
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,merge,LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import BatchNormalization,Reshape
from keras import optimizers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import layers


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json



from keras.models import load_model
from keras.models import model_from_json
loaded_merged_model=load_model('my_model_our_dataset2_revised.hdf5')

loaded_merged_model.summary()

import numpy as np
import cv2 as cv
import os
import urllib
import urllib.request
import numpy as np
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import collections

count=np.zeros((5,1))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img=cv2.imread('new.jpg')

import matplotlib.pyplot as plt
plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray.shape

FACIAL_LANDMARKS_IDXS = collections.OrderedDict([("right_eye",(36, 42)),("left_eye",(42, 48))])

rects = detector(gray, 1)

rects

for(i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
	# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
			clone = img.copy()
			
				
			if name=="left_eye":
				(x,y)=shape[42]
				left_left_x=x
				left_left_y=y
				(x, y)=shape[45]
				left_right_x=x
				left_right_y=y
			
					
		
	
		
		left=img[(left_left_y-20):(left_right_y+20),(left_left_x-5):(left_right_x+5)]
		
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
			clone = img.copy()
			if name=="right_eye":
				(x,y)=shape[36]
				left_left_x1=x
				left_left_y1=y
				(x, y)=shape[39]
				left_right_x1=x
				left_right_y1=y
			
					
		
		
		right=img[(left_left_y1-20):(left_right_y1+20),(left_left_x1-5):(left_right_x1+5)]

left=cv.cvtColor(left, cv2.COLOR_BGR2GRAY)
left=cv.resize(left,(64,64))
right=cv.cvtColor(right, cv2.COLOR_BGR2GRAY)
right=cv.resize(right,(64,64))
left=left.astype('float32')
left=left/255
left=left.reshape(1,1,64,64)
right=right.astype('float32')
right=right/255
right=right.reshape(1,1,64,64)

prediction=[]
prediction = loaded_merged_model.predict([left,right],verbose=0)

pred_classes = np.argmax(prediction,axis=1)
if pred_classes==0:
		count[pred_classes]=count[pred_classes]+1
		print('center')

elif pred_classes==2:
		count[pred_classes]=count[pred_classes]+1
		print('right')
elif pred_classes==3:
		count[pred_classes]=count[pred_classes]+1
		print('left')
elif pred_classes==4:
		count[pred_classes]=count[pred_classes]+1
		print('up')
elif pred_classes==1:
		count[pred_classes]=count[pred_classes]+1
		print('down')


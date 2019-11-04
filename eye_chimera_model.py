# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""
Created on SAT FEB 16 07:07:05 2019

@author: subra
"""


#%%


# Import libraries

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

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import initializers



from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import ModelCheckpoint

#%%

#PATH = 'project'
# Define data path
#data_path = 'project'
#data_path='eye_chimera'




img_rows=64
img_cols=64
num_channel=3


# Define the number of classes
num_classes = 7
names = ['center','downleft','downright','left','right','upleft','upright']
data_path = 'eye_chimera\eyes\left_eyes'
data_dir_list = os.listdir(data_path)
img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
#%%
#img_data = img_data.astype('float32')
#img_data /= 255
print (img_data.shape)
num_channel=1
if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,1,1)
		print (img_data.shape)
		
left_train=img_data.astype('float32')
left_train=left_train/255
left_train=left_train.reshape(left_train.shape[0],1,img_rows,img_cols)


num_classes =7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

m1=319
m2=131
m3=124
m4=132
m5=138
m6=133
m7=137
labels[0:m1]=0
labels[m1:(m1+m2)]=1
labels[(m1+m2):(m1+m2+m3)]=2
labels[(m1+m2+m3):(m1+m2+m3+m4)]=3
labels[(m1+m2+m3+m4):(m1+m2+m3+m4+m5)]=4
labels[(m1+m2+m3+m4+m5):(m1+m2+m3+m4+m5+m6)]=5

labels[(m1+m2+m3+m4+m5+m6):(m1+m2+m3+m4+m5+m6+m7)]=6

names = ['center','downleft','downright','left','right','upleft','upright'
			]

#%%

	  
# convert class labels to on-hot encoding
Y_train = np_utils.to_categorical(labels, num_classes)


#############################
names = ['center','downleft','downright','left','right','upleft','upright']
data_path = 'right_eyes'
data_dir_list = os.listdir(data_path)
img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
#%%
#img_data = img_data.astype('float32')
#img_data /= 255
print (img_data.shape)
num_channel=1
if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,1,1)
		print (img_data.shape)
		
right_train=img_data.astype('float32')
right_train=right_train/255
right_train=right_train.reshape(right_train.shape[0],1,img_rows,img_cols)

right_train1, right_test, Y_train1, Y_test = train_test_split(right_train,Y_train, test_size=0.2)
#%%
left_train1,left_test,Y_train1, Y_test =train_test_split(left_train,Y_train,test_size=.2)



input_shape_video=(1,img_rows,img_cols)

input2=Input(shape=input_shape_video)
input3=Input(shape=input_shape_video)

x = Conv2D(32, (7,7), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=0.01))(input2)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#x = BatchNormalization()(x)
#x1=TimeDistributed(Flatten())(x)
#x1=TimeDistributed(Dense(64,activation='relu'))(x1)


#x = TimeDistributed(ZeroPadding2D((3, 3)))(input2)

x = Conv2D(64, (7,7), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=0.01))(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)


x = Conv2D(128, (7,7), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=0.01))(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

z1=Flatten()(x)


x = Conv2D(32, (7,7), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=0.01))(input3)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#x = BatchNormalization()(x)
#x1=TimeDistributed(Flatten())(x)
#x1=TimeDistributed(Dense(64,activation='relu'))(x1)


#x = TimeDistributed(ZeroPadding2D((3, 3)))(input2)


x = Conv2D(64, (7,7), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=0.01))(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)


x = Conv2D(128, (7,7), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=0.01))(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

z2=Flatten()(x)

x5 = merge.concatenate([z1,z2], axis=-1)
out=Dense(7,activation='softmax', kernel_initializer=initializers.random_normal(stddev=0.01))(x5)










model=Model(inputs=[input2,input3],outputs=out)

model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])
num_epoch=100
#y15=Activation('relu')(y14)
#from keras.models import load_model
#from keras.models import model_from_json
#loaded_merged_model=load_model('eye_gaze/ezt/msdensenet_on_skin_epoch5.hdf5')
#%%
#merged_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
#merged_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

#%%activate tensorflow-gpu2

#checkpoint=[ModelCheckpoint(filepath='eye_gaze/ezt/my_model_hpeg.hdf5',  monitor='val_acc',save_best_only=True, mode='max',verbose=1)]
#batch_size=10
#hist = model.fit(datagen.flow(X_train,y_train,batch_size=batch_size),steps_per_epoch=len(x)/batch_size, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
#hist = model.fit([left_train,right_train], Y_train, batch_size=batch_size, epochs=20, verbose=1,callbacks=checkpoint, validation_split=.1)

checkpoint=[ModelCheckpoint(filepath='my_model_chimera2.hdf5',  monitor='val_acc',save_best_only=True, mode='max',verbose=1 )]
batch_size=32
#hist = model.fit(datagen.flow(X_train,y_train,batch_size=batch_size),steps_per_epoch=len(x)/batch_size, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
hist = model.fit([left_train1,right_train1], Y_train1, batch_size=batch_size, epochs=num_epoch, verbose=1,callbacks=checkpoint,validation_split=.1)



score=model.evaluate([left_test,right_test], Y_test,verbose=1)
print(score) 

from keras.models import load_model
from keras.models import model_from_json
loaded_merged_model=load_model('my_model_chimera2.hdf5')
#score=loaded_merged_model.evaluate([left_test,right_test], Y_test,verbose=1)
#print(score) 

# Viewing model_configuration
#%%
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("resnet_lstm_epoch1.h5")
#print("Loaded model from disk")
#loaded_model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])

#%%
#checkpoint=[ModelCheckpoint(filepath='resnet50_lstm_epoch1.hdf5')]
#batch_size=10
#hist = model.fit(datagen.flow(X_train,y_train,batch_size=batch_size),steps_per_epoch=len(x)/batch_size, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
#hist =loaded_model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1, validation_data=(X_dev, y_dev))

#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)


#%%




# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%

# Evaluating the model




#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = loaded_merged_model.predict([left_test,right_test])
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = names
					
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float16') / cm.sum(axis=1)[:, np.newaxis] 
        #cm=float("{0:.2f}".format(cm))
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #cm=float("{0:.2f}".format(cm))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float("{0:.2f}".format(cm[i,j])),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=12,fontstyle='normal')
    plt.xlabel('Predicted label',fontsize=12,fontstyle='normal')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names,
 #                     title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,normalize=True
                      )
#plt.figure()
plt.savefig('eye_chimera_conf_matrix_final.eps',format='eps',dpi=2000)
plt.show()

#%%
# Saving and loading model and weights
#from keras.models import model_from_json
#from keras.models import load_model

# serialize model to JSON
#model_json = merged_model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#merged_model .save_weights("msdensenet_on_skin_epoch5.h5")
#print("Saved model to disk")

# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

#merged_model .save('msdensenet_on_skin_epoch5.h5')
#loaded_model=load_model('model.hdf5')



# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-




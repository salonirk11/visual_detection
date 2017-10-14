# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import libaries
import pandas as pd
import numpy as np
import cv2
import os, sys
from math import ceil
from tqdm import tqdm
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256,256))
    return img

def show(X):
    cv2.imshow('img',X)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def pad(img):
    a=np.full((256,256,3),128,dtype=int)
    k=img.shape[0]/2
    l=img.shape[1]/2
    x=ceil(128-k)
    y=ceil(128+k)
    z=ceil(128-l)
    w=ceil(128+l)
    a[x:y,z:w,:]=img
    return np.array(a).astype('uint8')

def gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def mean_threshold(img):
    mask=cv2.adaptiveThreshold(gray(img),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return cv2.bitwise_and(img,img,mask=mask)

def gaussian_threshold(img):
    mask=cv2.adaptiveThreshold(gray(img),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return cv2.bitwise_and(img,img,mask=mask)

def adjust_gamma(image):
    gamma=1.5
    invGamma=1.0/gamma
    table=np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype('uint8')
    image=cv2.LUT(image, table)
    return image

TRAIN_PATH = 'train_img/'
TEST_PATH = 'test_img/'


images=train['image_id'].values
labels=train['label'].values

train_img, test_img = [],[]
label_list=[]

def img_process(i):
    image = cv2.imread(TRAIN_PATH+labels[i]+'/'+images[i]+'.png', cv2.IMREAD_COLOR)
    
    image=adjust_gamma(image)
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 10, 250)

    (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    idx = 0  
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>70 and h>70:
            idx+=1
            new_img=image[y:y+h,x:x+w]
            new_img=gaussian_threshold(new_img)
            new_img=pad(new_img)
            
            train_img.append(new_img)
            label_list.append(labels[i])
     
        #cv2.imwrite(TRAIN_PATH+labels[i]+'/'+images[i]+'_'+ str(idx) +'.png', new_img)
   

for i in range(len(images)):
    img_process(i)


for img_path in tqdm(test['image_id'].values):
    test_img.append(adjust_gamma(read_img(TEST_PATH + img_path + '.png')))

x_train = np.array(train_img, np.float32) / 255.
x_test = np.array(test_img, np.float32) / 255.


Y_train = {k:v+1 for v,k in enumerate(set(label_list))}
y_train = [Y_train[k] for k in label_list]   
y_train = np.array(y_train)

y_train = to_categorical(y_train)

#model.summary()

batch_size = 8 
epochs = 15 

train_datagen = ImageDataGenerator(
        rotation_range=20,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='constant'
)

train_datagen.fit(x_train)

base_model = applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
              metrics=['accuracy'])

history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('INCEPTION_RESNET_V2.model', monitor='val_acc', save_best_only=True)]
)

predictions = model.predict(x_test)

predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in Y_train.items()}
pred_labels = [rev_y[k] for k in predictions]


sub = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels})
sub.to_csv('sub_vgg1.csv', index=False) ## ~0.59

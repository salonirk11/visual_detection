import pandas as pd
import os

train = pd.read_csv('train.csv')

TRAIN_PATH = 'train_img/'

#label_list = set(train['label'].values)
#for i in label_list:
#    os.makedirs(TRAIN_PATH+i)


images=train['image_id'].values
labels=train['label'].values

data=[]
for i in range(len(images)):
    x=[]
    x.append(images[i])
    x.append(labels[i])
    data.append(x)

for x in (data):
    os.rename(TRAIN_PATH+x[0]+'.png', TRAIN_PATH+x[1]+'/'+x[0]+'.png')

#print(label_list)


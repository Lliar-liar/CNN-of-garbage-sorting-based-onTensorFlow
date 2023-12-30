#-*-coding:gb2312#-*-
from distutils.errors import PreprocessError
from pickle import FALSE, NONE
from pickletools import optimize
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import matplotlib.pyplot as plt
import json
from keras.applications import VGG16 
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os
import cv2

testnow=0
train_dir = "D:\\data\\garbage_classify\\train_data"  
test_dir = "D:\\data\\garbage_classify_et\\train_data"    
height=200
lenth=200
input_shape=(height,lenth,3)
input_shape1=(-1,height,lenth,3)
resize_shape=(height,lenth)
conv_base = VGG16(weights='imagenet',      
                  include_top=False,        
                  input_shape=input_shape) 


model = models.Sequential()             
model.add(conv_base)             #VGG16模型，不使用原始的分类器
model.add(layers.Flatten())  #用于全连接层和卷积层的连接了，将输出张量展平，便于连接之后的Dense层      
model.add(layers.Dense(units=512,activation='relu'))
#model.add(layers.Dense(units=512,activation='relu'))#层数为512的全连接层
#model.add(layers.Dropout(0.3)) 
model.add(layers.Dense(units=40, activation='softmax'))  
conv_base.trainable=False    #将VGG16设置为不可训练状态，减小计算的参数量
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

if(testnow==1):
    model.load_weights('D:\\data\\garbage_classify_et\\model.h5')
    test_img=cv2.imread('D:\\data\\garbage_classify_et\\t.jpg')
    test_img=cv2.resize(test_img,resize_shape)
    test_img=test_img.reshape(input_shape1)
    result=model.predict(test_img)
    result=np.argmax(result,axis=1)
    dict={
    "0": "可回收物/一次性快餐盒",
    "1": "干垃圾/污损塑料",
    "2": "干垃圾/烟蒂",
    "3": "干垃圾/牙签",
    "4": "干垃圾/破碎花盆及碟碗",
    "5": "干垃圾/竹筷",
    "6": "湿垃圾/剩饭剩菜",
    "7": "干垃圾/大骨头",
    "8": "湿垃圾/水果果皮",
    "9": "湿垃圾/水果果肉",
    "10": "湿垃圾/茶叶渣",
    "11": "湿垃圾/菜叶菜根",
    "12": "湿垃圾/蛋壳",
    "13": "湿垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}
    print (dict['%d'%result])
    exit(0)
'''
def data_preprocess_t(data):
    data=data.copy()
    print(data.columns)
    #y=data.pop('label')
    x=data.values
    print(x.shape)
    #y=np.array(y)
    x.reshape(-1,28,28,1)
    
    return x
'''
#train= pd.read_csv(train_dir)

label=[]
test_label=[]
train_data=[]
test_data=[]
for i in range(0,19735):
    #print(os.path.exists(train_dir+"\\fimg_%d.txt"%i))
    if(os.path.exists(train_dir+"\\img_%d.txt"%i)==False):
        continue
    f=open(train_dir+"\\img_%d.txt"%i,'r')
    fi=f.readlines()
    train_image=cv2.imread(train_dir+"\\img_%d.jpg"%i)
    #print(type(train_image)==np.ndarray)
    if(type(train_image)!=np.ndarray):
        continue
    #print(train_image.shape)
    train_image=cv2.resize(train_image,resize_shape)
    train_image=train_image.reshape(input_shape1)
    train_data.append(train_image)
    for j in fi:
        j=j.strip('\n').split(",")
        #print(j[1])
        label.append(int(j[-1]))
label=np.array(label)
train_images = np.concatenate(train_data, axis=0)
train_labels=to_categorical(label)

for i in range(0,5000):
    #print(os.path.exists(train_dir+"\\fimg_%d.txt"%i))
    if(os.path.exists(test_dir+"\\fimg_%d.txt"%i)==False):
        continue
    f=open(test_dir+"\\fimg_%d.txt"%i,'r')
    fi=f.readlines()
    test_image=cv2.imread(test_dir+"\\fimg_%d.jpg"%i)
    #print(type(train_image)==np.ndarray)
    if(type(test_image)!=np.ndarray):
        continue
    #print(train_image.shape)
    test_image=cv2.resize(test_image,resize_shape)
    test_image=test_image.reshape(input_shape1)
    test_data.append(test_image)
    for j in fi:
        j=j.strip('\n').split(",")
        #print(j[1])
        test_label.append(int(j[-1]))
test_label=np.array(test_label)
test_images = np.concatenate(test_data, axis=0)
test_labels=to_categorical(test_label)
#plt.imshow(train_images[0])
#plt.show()
#print (test_labels.shape)
#print (test_images.shape)

def group_split(X, y, train_size=0.8):
    splitter = ShuffleSplit(train_size=train_size)  
    train, test = next(splitter.split(X, y))  
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

#train_images,test_images,train_labels,test_labels=group_split(train_images, train_labels, train_size=0.8)

model.fit(train_images,train_labels,validation_split=0.15,batch_size=64,epochs=5,
       callbacks=None)
loss, acc = model.evaluate(test_images, test_labels)
print("train model, accuracy:{:5.2f}%".format(100 * acc))
model.save_weights('D:\\data\\garbage_classify_et\\model.h5')


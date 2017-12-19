#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 23:00:45 2017

@author: yangzhan
"""
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.platform import gfile
import os


modeldir = './model/20170512-110547.pb' #change to your model dir
print('建立facenet embedding模型')
tf.Graph().as_default()

with gfile.FastGFile(modeldir,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
print('facenet embedding模型建立完毕')

#%%
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y 

def img_to_encoding(image_path,sess,image_size=200):
    image = cv2.imread(image_path)
    image = image[:, :, (2, 1, 0)]
    image=cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image=prewhiten(image)
    scaled_reshape = []
    scaled_reshape.append(image.reshape(-1,image_size,image_size,3))
    emb_array = np.zeros((1, embedding_size))
    emb_array[0, :] = sess.run(embeddings, feed_dict=
              {images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]
    return emb_array

def who_is_it(image_path, database,sess):
    encoding = img_to_encoding(image_path,sess)
    min_dist = 2
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc-encoding)
        print(name,dist)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity
#%%
image_size=200
sess = tf.Session()
database={}
file_path='./images/'
for image in os.listdir(file_path):
    name=image.split('.')[0]
    print(file_path+image)
    database[name]=img_to_encoding(file_path+image,sess,image_size)
#%%
print(who_is_it('1.jpg',database,sess))
    
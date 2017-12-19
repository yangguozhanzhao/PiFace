# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:48:23 2017

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

def img_to_encoding(image,sess,image_size=200):
	if type(image)==str:
		image = cv2.imread(image)
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
image_size=150
sess = tf.Session()
database={}
file_path='./images/'
for image in os.listdir(file_path):
	name=image.split('.')[0]
	print(file_path+image)
	database[name]=img_to_encoding(file_path+image,sess,image_size)
# print(who_is_it('1.jpg',database,sess))
	
"""
获取屏幕内的用户头像，作为对比
""" 
from picamera.array import PiRGBArray
from picamera import PiCamera
 
import time
import cv2
import os
import pygame
 
os.putenv( 'SDL_FBDEV', '/dev/fb1' )
 
# Setup the camera

camera = PiCamera()
camera.resolution = ( 500, 500 )
camera.framerate = 30
rawCapture = PiRGBArray( camera, size=( 500, 500 ) )
 
fcounter = 0
facefind = 0
 
# Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier( 'lbpcascade_frontalface.xml' )
t_start = time.time()
fps = 0
 
### Main ######################################################################

# Capture frames from the camera
for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):

	image = frame.array
	# Run the face detection algorithm every four frames
	if fcounter == 12:
		fcounter = 0
		# Look for faces in the image using the loaded cascade file
		gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
		faces = face_cascade.detectMultiScale( gray )
		print "Found " + str( len( faces ) ) + " face(s)"
		
		#检测到头像
		if str( len( faces ) ) != 0:
			facefind = 1
			facess = faces
		else:
			facefind = 0
		# Draw a rectangle around every face
		for ( x, y, w, h ) in faces:
			k=max(w,h)
			if k>image_size:
				cv2.rectangle( image, ( x, y ), ( x + k, y + k ), ( 200, 255, 0 ), 1)
				face_image=image[y:y+k,x:x+k,:]
				_,name=who_is_it(face_image,database,sess)
				cv2.putText( image,  name, ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 1 )
			break 
	fcounter += 1
 
 
	# Calculate and show the FPS
	fps = fps + 1
	sfps = fps / ( time.time() - t_start )
	cv2.putText( image, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 255, 255 ), 2 )
 
	cv2.imshow( "Frame", image )
	cv2.waitKey( 1 )
 	rawCapture.truncate( 0 )

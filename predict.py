# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:48:23 2017

@author: yangzhan
"""
import os
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import time

def triplet_loss(y_true, y_pred, alpha = 0.2):
	anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
	# Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
	pos_dist = tf.reduce_sum(tf.square(anchor-positive),axis=-1)
	# Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
	neg_dist = tf.reduce_sum(tf.square(anchor-negative),axis=-1)
	# Step 3: subtract the two previous distances and add alpha.
	basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
	# Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
	loss = tf.reduce_sum(tf.maximum(basic_loss,0))
	### END CODE HERE ###    
	return loss

def who_is_it(image_path, database, model):
	## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
	encoding = img_to_encoding(image_path,FRmodel)
	## Step 2: Find the closest encoding ##
	# Initialize "min_dist" to a large value, say 100 (≈1 line)
	min_dist = 100
	# Loop over the database dictionary's names and encodings.
	for (name, db_enc) in database.items():
		# Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
		dist = np.linalg.norm(db_enc-encoding)
		# If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
		if dist < min_dist:
			min_dist = dist
			identity = name    
	if min_dist > 0.7:
		print("Not in the database.")
		print ('min_dist='+str(min_dist))
	else:
		print ("it's " + str(identity) + ", the distance is " + str(min_dist))
		
	return min_dist, identity

def generate_database(file_path):
	database={}
	for image in os.listdir(file_path):
		name = image.split('.')[0]
		database[name] = img_to_encoding(file_path+image, FRmodel)
	return database

start=time.time()
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params()) 
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
load_weights_time=time.time()
print("load weight:",load_weights_time-start)

database=generate_database('./images/')
database_time=time.time()
print("database:",database_time-load_weights_time)
print(len(database))
#0.5441
#~ who_is_it('1.jpg',database,FRmodel)
#~ predict_time=time.time()
#~ print("predict:",predict_time-database_time)

#coding=utf-8
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
camera.resolution = ( 320, 240 )
camera.framerate = 30
rawCapture = PiRGBArray( camera, size=( 320, 240 ) )
 
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
	if fcounter == 6:
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
			cv2.rectangle( image, ( x, y ), ( x + k, y + k ), ( 200, 255, 0 ), 1)
			face_image=image[y:y+k,x:x+k,:]
			_,name=who_is_it(face_image,database,FRmodel)
			#cv2.putText( image,  name+u"你好 ", ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 1 )
			break
			#
			#facess = faces
			#~ 
	#~ else:
		#~ if facefind == 1 and str( len( facess ) ) != 0:
			#~ # Continue to draw the rectangle around every face
			#~ for ( x, y, w, h ) in facess:
				#~ cv2.rectangle( image, ( x, y ), ( x + w, y + h ), ( 200, 255, 0 ), 1 )
				#~ cv2.putText( image, "Face No." + str( len( facess ) ), ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 1 )
 
	fcounter += 1
 
 
	# Calculate and show the FPS
	fps = fps + 1
	sfps = fps / ( time.time() - t_start )
	cv2.putText( image, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 255, 255 ), 2 )
 
	cv2.imshow( "Frame", image )
	cv2.waitKey( 1 )
 	rawCapture.truncate( 0 )

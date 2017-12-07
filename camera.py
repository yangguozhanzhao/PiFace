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
			cv2.rectangle( image, ( x, y ), ( x + w, y + h ), ( 200, 255, 0 ), 1)
			face_image=image[y:y+h,x:x+w,:])
			
			break
			#cv2.putText( image, "Face No." + str( len( facess ) ), ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 1 )
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

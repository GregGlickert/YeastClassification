# import the necessary packages
import argparse

import cv2
from plantcv import plantcv as pcv
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
try:
	refPt = []
	cropping = False
	def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, cropping
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			refPt = [(x, y)]
			cropping = True
		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			refPt.append((x, y))
			cropping = False
			# draw a rectangle around the region of interest
			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", image)


	image = cv2.imread("P53 Plate A copy.JPG")
	image = cv2.resize(image, (1200,800))
	cv2.imwrite("rezised.png", image)
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		#cv2.imshow("ROI", roi)
		#cv2.waitKey(0)
	centroids_x = ((refPt[0][0] + refPt[1][0])/2) * 5
	centroids_y = ((refPt[0][1] + refPt[1][1])/2) * 5
except:
	pass
print(centroids_x)




# close all open windows


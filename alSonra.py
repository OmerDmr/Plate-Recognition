import scipy.misc
import pickle
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from Features import *
from visualizations import *
from PIL import Image

Heatmap_buffer = []
buffer_weights=[0.1,0.2,0.3,0.4]
heatmap_buffer = []
N_buffer = 3
search_window_scales = (1,1.5, 2) # (256x64), (384x96), (512x128)


car = cv2.imread('01.jpeg')


filename='pltClassifier.p'
with open(filename, 'rb') as file:
	clsModel = pickle.load(file)
	model = clsModel["model"]
	X_scaler = clsModel["scaler"]

def slidingWindows(img, scales, xy_window=(256, 64)):


	wind0=xy_window[0]
	wind1=xy_window[1]
	over = 1
	window_list = []
	sc = 0
	while sc <= 2:
		xs=0
		wind0 = np.int(wind0*over)
		wind1 = np.int(wind1*over)
		while xs + wind1 <= img.shape[1]:
			ys=0
			while ys + wind0 <= img.shape[0]:
				starty = ys
				startx = xs
				endx = wind1+xs
				endy = wind0+ys
				ys = ys + 16
				window_list.append(((starty, endy), (startx, endx)))
				print (((starty, endy), (startx, endx)))
			xs = xs+16
		over=over+0.5
		wind0=xy_window[0]
		wind1=xy_window[1]
		sc = sc+1
				

	return window_list

def searchWindows(img, scales, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32, orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

	img = img.astype(np.float32)/255
	
	if  color_space == 'YCrCb':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	windows=slidingWindows(img,scales)

	on_windows = [] 
	for window in windows:
		starty = int(window[0][0])
		endy = int(window[0][1])
		startx = int(window[1][0])
		endx = int(window[1][1])
		
		print((endx-startx),(endy-starty))
		subimg = cv2.resize(img[window[0][0]:window[0][1], window[1][0]:window[1][1]], (256, 64))  # training image is (256,64)
		print(subimg.shape)
		features = extractingImgFeatures(subimg, color_space=color_space,
		                        spatial_size=spatial_size, hist_bins=hist_bins,
		                        orient=orient, pix_per_cell=pix_per_cell,
		                        cell_per_block=cell_per_block,
		                        hog_channel=hog_channel, spatial_feat=spatial_feat,
		                        hist_feat=hist_feat, hog_feat=hog_feat)
		features = np.resize(features,(3924,1))
		X = np.array(features).reshape(1, -1)
		test_features = scaler.transform(X)
		print(test_features)
		prediction = clf.predict(test_features)
		print(prediction)
		if prediction == 1:
			on_windows.append(window)
			cv2.rectangle(img, (startx, starty), (endx, endy), (0, 0, 255), 6)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

	imcopy = np.copy(img)

	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

	return imcopy


def add_heat(heatmap, bbox_list):
	for box in bbox_list:
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	return heatmap


def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	
	return heatmap


def draw_bboxes(img, heatmap_buffer, heatmap_pre, N_buffer):

	heatmap_buffer.append(heatmap_pre)

	if len(heatmap_buffer) > N_buffer:
		heatmap_buffer.pop(0)


	idxs = range(N_buffer)
	for b, w, idx in zip(heatmap_buffer, buffer_weights, idxs):
		heatmap_buffer[idx] = b * w

	heatmap = np.sum(np.array(heatmap_buffer), axis=0)
	heatmap = apply_threshold( heatmap, threshold= sum(buffer_weights[0:N_buffer])*2)

	labels = label(heatmap)

	bboxes = []
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		bbox_tmp = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		bboxes.append(bbox_tmp)


	for bbox in bboxes:
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)

	return img, heatmap, bboxes


def generate_heatmap(image, windows_list):

	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	heat = add_heat(heat, windows_list)
	heat = apply_threshold(heat, 1)
	heatmap = np.clip(heat, 0, 255)

	return heatmap




def plateDetect(image):

	image = cv2.resize(image,(1000,1000))

	start = timer()
	ystarts = [0, 0]
	ystops = [image.shape[0], image.shape[0]]

	windows_list = searchWindows(np.copy(image), search_window_scales, model,X_scaler, orient, pix_per_cell, cell_per_block,
								spatial_size, hist_bins, hog_channel, color_space, spatial_feat, hist_feat, hog_feat)
	"""
	heatmap_pre = generate_heatmap(image, windows_list)

	draw_img, heatmap_post, bboxes = draw_bboxes(np.copy(image), copy(Heatmap_buffer), heatmap_pre, min(len(Heatmap_buffer)+1,N_buffer) )

	if len(Heatmap_buffer) >= N_buffer:
		Heatmap_buffer.pop(0)

	fps = 1.0 / (timer() - start)

	draw_img = draw_background_highlight(image, draw_img, image.shape[1])
	draw_thumbnails(draw_img, image, bboxes)"""

	return draw_img

drawingImage=plateDetect(car)
#drawingImage=cv2.resize(drawingImage,(700,700))

#cv2.imshow('image',drawingImage)
#cv2.waitKey(0)









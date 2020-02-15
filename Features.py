import matplotlib.image as mpimg
import matplotlib.pyplot as mplot
from skimage.feature import hog
from scipy.ndimage.measurements import label
import os
import numpy as np
import cv2
from copy import copy

### Parameters
color_space = 'YCrCb' 
orient = 9  
pix_per_cell = 8 
cell_per_block = 2
hog_channel = "ALL" 
spatial_size = (32, 32)
hist_bins = 32  
spatial_feat = True
hist_feat = True 
hog_feat = True 
feature_vec=False

def binSpatial(img, size=(32, 32)):
	color1 = cv2.resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	sptlFeatures = np.hstack((color1, color2, color3))
	return sptlFeatures
		                
def colorHist(img, nbins=32, binsRange=(0, 256)):
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=binsRange)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=binsRange)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=binsRange)
	histFeatures = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	return histFeatures

def getHogFeatures(img, orient, pix_per_cell, cell_per_block, vis=True):
	feature_vec=True
	if vis==True:
		features,hog_image = hog(img, orientations=orient, 
						pixels_per_cell=(pix_per_cell, pix_per_cell),
						cells_per_block=(cell_per_block, cell_per_block),
						transform_sqrt=True, 
						visualise=vis, feature_vector=feature_vec)
		return features,hog_image
	else:
		features = hog(img, orientations=orient, 
					pixels_per_cell=(pix_per_cell, pix_per_cell),
					cells_per_block=(cell_per_block, cell_per_block), 
					transform_sqrt=True, 
					visualise=vis, feature_vector=feature_vec)
		return features
def colorConvert(img, color_space='RGB'):
	if color_space == 'HSV':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	elif color_space == 'LUV':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
	elif color_space == 'HLS':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	elif color_space == 'YUV':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	elif color_space == 'YCrCb':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	return img

def extractFeature(images, color_space='RGB',spatial_size=(32, 32), hist_bins=32,
			orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
			spatial_feat=True, hist_feat=True, hog_feat=True):
	imgsFeatures = []
	for filename in images:
		print (filename)
		filenameFeature = []
		image=mpimg.imread(filename)
		image = cv2.resize(image,(256,64))
		if color_space != 'RGB':
			featureImg=colorConvert(image,color_space)
		else:
			featureImg=np.copy(image)
		if hist_feat == True:
			histFeatures=colorHist(featureImg)
			filenameFeature.append(histFeatures)
		if spatial_feat == True:
			sptlFeatures=binSpatial(featureImg, size=spatial_size)
			filenameFeature.append(sptlFeatures)
		if hog_feat == True:
			if hog_channel == 'ALL':
				hogFeatures = []
				for channel in range(featureImg.shape[2]):
					hogFeatures.append(getHogFeatures(featureImg[:,:,channel],
					orient,pix_per_cell,cell_per_block,
					vis=False))
				hogFeatures=np.ravel(hogFeatures)
			else:
				hogFeatures=getHogFeatures(featureImg[:,:,hog_channel],orient,
						pix_per_cell,cell_per_block,vis=False)
			filenameFeature.append(hogFeatures)
		imgsFeatures.append(np.concatenate(filenameFeature))
	return imgsFeatures

def extractingImgFeatures(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
	imgFeatures = []
	if color_space != 'RGB':
		featureImg=colorConvert(img,color_space)
	else:
		featureImg = np.copy(img)

	if spatial_feat == True:
		sptlFeatures = binSpatial(featureImg, size=spatial_size)
		imgFeatures.append(sptlFeatures)

	if hist_feat == True:
		histFeatures = colorHist(featureImg, nbins=hist_bins)
		imgFeatures.append(histFeatures)
  
		if hog_feat == True:
			if hog_channel == 'ALL':
				hogFeatures = []
				for channel in range(featureImg.shape[2]):
					hogFeatures.extend(getHogFeatures(featureImg[:,:,channel],
										orient,pix_per_cell,cell_per_block,
										vis=False))
			else:
				hogFeatures=getHogFeatures(featureImg[:,:,hog_channel],orient,
					pix_per_cell,cell_per_block,vis=False)
		imgFeatures.append(hogFeatures)

	return np.concatenate(imgFeatures)

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

"""
def slidingWindows(image,model,X_scaler):
	image = cv2.resize(image,(1200,1200))
	image = image.astype(np.float32)/255
	tmp = image 
	stepSize = 32
	(w_width, w_height) = (32, 16)
	windowList = [] 
	for x in range(0, image.shape[1] - w_width , stepSize):
		for y in range(0, image.shape[0] - w_height, stepSize):
			window = image[x:x + w_width, y:y + w_height]
			windowFeatures = extractingImgFeatures(window)
			windowFeatures = np.resize(windowFeatures,(16128,1))
			print (windowFeatures)
			X = np.array(windowFeatures).reshape(1, -1)
			features = X_scaler.transform(X)
			print 	(features.shape)
			prediction = model.predict(features)
			if prediction == 1:
				cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0, 0, 255), 6)
				windowList.append(window)
				plt.imshow(np.array(tmp).astype('uint8'))      			
	plt.show()
	return windowList
"""




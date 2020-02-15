import PIL
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
from skimage.feature import hog
from skimage import data,exposure

class featExt:

	def __init__(self,img):
		self.img=img

	def binSpatial(self, img, size=(64, 64)):

	    color1 = cv2.resize(img[:,:,0], size).ravel()
	    color2 = cv2.resize(img[:,:,1], size).ravel()
	    color3 = cv2.resize(img[:,:,2], size).ravel()
	    sptlFeatures = np.hstack((color1, color2, color3))
	    return sptlFeatures
		                
	def colorHist(self, img, nbins=64, binsRange=(0,256)):
	 
	    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=binsRange)
	    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=binsRange)
	    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=binsRange)
	    histFeatures = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	    return histFeatures


	def getHogFeatures(self,img, orient, pix_per_cell, cell_per_block, vis=True, feature_vector=True):

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

	def colorConvert(self, img, color_space='RGB'):
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


	def extractFeature(self, image, color_space='RGB',spatial_size=(64, 64), hist_bins=64,
				 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
				 spatial_feat=True, hist_feat=True, hog_feat=True):

		imgFeatures = []
		if color_space != 'RGB':
			featureImg=self.colorConvert(image,color_space)
		else:
			featureImg=np.copy(image)
		if hist_feat == True:
			histFeatures=self.colorHist(fetureImg,hist_bins)
			imgFeatures.append(histFeatures)
		if spatial_feat == True:
			sptlFeatures=self.binSpatial(fetureImg,spatial_size)
			imgFeatures.append(sptlFeatures)
		if hog_feat == True:
			if hog_channel == 'ALL':
				hogFeatures = []
				for channel in range(featureImg.shape[2]):
					hogFeatures.append(self.getHogFeatures(featureImg[:,:,channel],
					orient,pix_per_cell,cell_per_block,
					vis=False, feature_vec=True))
				hogFeatures=np.ravel(hogFeatures)
			else:
				hogFeatures=self.getHogFeatures(featureImg[:,:,hog_channel],orient,
						pix_per_cell,cell_per_block,vis=False,feture_vec=True)
			imgFeatures.append(hogFeatures)
		return imgFeatures












import PIL
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
from skimage.feature import hog
from skimage import data,exposure
#from featureExtraction import extractFeature
#from featureExtraction import colorHist,binSpatial
#from featureExtraction import getHogFeatures




validImages=['.jpg','.jpeg','.png']

class send:
	def __initSend__(self,img,*atr):
		self.img=img
		self.atr= []
	print('\nevet')
	def extract(self):
		pathCar='/home/kafein/plateRecognition/carImages'
		carFeatures = []
		for filename in os.listdir(pathCar):
			ext = os.path.splitext(filename)[1]
	    		if ext.lower() not in validImages:
				continue
			image=Image.open(os.path.join(pathCar,filename))
			image=image.resize((64,64),PIL.Image.ANTIALIAS)
		    	features = self.extractFeature(image, color_space=self.color_space, 
		                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
		                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
		                        cell_per_block=self.cell_per_block, 
		                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
		                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
		    	carFeatures.append(np.concatenate(features))

		

		pathNotCar='/home/kafein/plateRecognition/notCarImages'
		notCarFeatures = []
		for filename in os.listdir(pathNotCar):
			ext = os.path.splitext(filename)[1]
	    		if ext.lower() not in validImages:
				continue
			image=Image.open(os.path.join(pathNotCar,filename))
			image=image.resize((64,64),PIL.Image.ANTIALIAS)
			features = self.extractFeature(image, color_space=self.color_space, 
		                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
		                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
		                        cell_per_block=self.cell_per_block, 
		                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
		                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
		    	notCarFeatures.append(np.concatenate(features))

img=send()
img.extract()

		


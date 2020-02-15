import cv2
import scipy.misc
from skimage.filters import threshold_otsu
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from os import walk
from os import path
import time
import pickle



car = cv2.imread('car.jpg')

print(car.shape)

grayCar = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
grayCar = scipy.misc.imresize(grayCar, 0.3)

print(grayCar.shape)

cv2.imshow('gray',grayCar)
cv2.waitKey(0)

thresh = threshold_otsu(grayCar)
binary = grayCar > thresh
labeled = measure.label(binary)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(grayCar, cmap="gray")
ax2.imshow(binary, cmap="binary")

plt.show()




def labelImages(images):
	imgsFeatures = []
	for filename in images:
		print filename
		image=np.array(cv2.imread(filename))
		image = cv2.resize(image, (128, 32))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		threshold = threshold_otsu(image)
		binary = image < threshold
		labeled = binary.reshape(-1)
		imgsFeatures.append(labeled)
	return imgsFeatures

def get_fileNames(path):
	data = []
	for root, dirs, files in os.walk(path):
    		for name in files:
        		if name.endswith((".jpg", ".jpeg",".png")):
				baseName=os.path.join(root,name)
				print(baseName)
				data.append(baseName)
	return data



def trainAndTest():

	pathPlt='/home/kafein/plateRecognition/plates/plt'
	plates=get_fileNames(pathPlt)
	print(len(plates))
	plateFeatures=labelImages(plates)

	pathNoPlt='/home/kafein/plateRecognition/plates/notPlt'
	noPlates=get_fileNames(pathNoPlt)
	print(len(noPlates))
	noPlateFeatures=labelImages(noPlates)

	X = np.vstack((plateFeatures, noPlateFeatures)).astype(np.float64)                        
	X_scaler = RobustScaler().fit(X)
	X_scaler = MinMaxScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	y = np.hstack((np.ones(len(plateFeatures)), np.zeros(len(noPlateFeatures))))

	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	scaled_X, y, test_size=0.2, random_state=rand_state)

	t=time.time()
	svc=LinearSVC().fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	score1 = round(svc.score(X_test, y_test), 4) 
	print('Test Accuracy of SVC = ', score1)
	t=time.time()

	tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
	 					max_depth=3, min_samples_leaf=5)
	tree.fit(X_train, y_train)
	score2 = round(tree.score(X_test, y_test), 4)
	print('Test Accuracy of DecisionTree = ', score2)

	gauss = GaussianNB()
	gauss.fit(X_train, y_train)
	score3 = round(gauss.score(X_test, y_test), 4)
	print('TesT Accuracy of Gaussian = ', score3)

	if score1 > score2:
		if score1 > score3:
			select = score1
			print('Best Accuracy = SVC')
		else:
			select = score3
			print('Best Accuracy = Gaussian')
	else:
		if score2 > score3:
			select = score2
			print('Best Accuracy = DecisionTree')
		else:
			select = score3
			print('Best Accuracy = Gaussian')

	modelName = 'pltClsModel.p'
	clsModel = {}
	clsModel["model"] = select
    	clsModel["scaler"] = X_scaler
	pickle.dump(clsModel, open(modelName, 'wb'))
	print "model saved"

trainAndTest()




	









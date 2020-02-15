from os import walk
from os import path
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from timeit import default_timer as timer
from Features import *




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
	

	pathPlates='/home/kafein/plateRecognition/plates/plt'
	plates=get_fileNames(pathPlates)
	print(len(plates))
	pltFeatures = extractFeature(plates, color_space=color_space, 
			               spatial_size=spatial_size, hist_bins=hist_bins, 
			               orient=orient, pix_per_cell=pix_per_cell, 
			               cell_per_block=cell_per_block, 
			               hog_channel=hog_channel, spatial_feat=spatial_feat, 
			               hist_feat=hist_feat, hog_feat=hog_feat)

	pathNotPlt='/home/kafein/plateRecognition/plates/Extras'
	notPlt=get_fileNames(pathNotPlt)
	print(len(notPlt))
	notPltFeatures = extractFeature(notPlt, color_space=color_space, 
			               spatial_size=spatial_size, hist_bins=hist_bins, 
			               orient=orient, pix_per_cell=pix_per_cell, 
			               cell_per_block=cell_per_block, 
			               hog_channel=hog_channel, spatial_feat=spatial_feat, 
			               hist_feat=hist_feat, hog_feat=hog_feat)
		
	X = np.vstack((pltFeatures, notPltFeatures)).astype(np.float64)                        
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)
	y = np.hstack((np.ones(len(pltFeatures)), np.zeros(len(notPltFeatures))))

	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	     
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
			select = svc
			print('Best Accuracy = SVC')
		else:
			select = gauss
			print('Best Accuracy = Gaussian')
	else:
		if score2 > score3:
			select = tree
			print('Best Accuracy = DecisionTree')
		else:
			select = gauss
			print('Best Accuracy = Gaussian')

	modelName = 'pltClassifier.p'
	clsModel = {}
	clsModel["model"] = select
	clsModel["scaler"] = X_scaler
	pickle.dump(clsModel, open(modelName, 'wb'))
	print ("model saved")
	
trainAndTest()

		


		


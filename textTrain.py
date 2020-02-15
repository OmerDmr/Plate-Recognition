import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def labeledText(textPath):
    	imageData = []
    	targetData = []
    	for letter in letters:
       		for filename in range(10):
            		imgPath = os.path.join(textPath, letter, letter + '_' + str(filename) + '.jpg')
            		image=imread(imgPath, as_grey=True)
			threshold = threshold_otsu(image)
			binary = image < threshold
			labeled = binary.reshape(-1)
            		imageData.append(labeled)
            		targetData.append(letter)

    	return (np.array(imageData), np.array(targetData))

def trainAndTest(model, num_of_fold, data, label):
    	accuracy = cross_val_score(model, data, label,
                                      	cv=num_of_fold)
    	print("Cross Validation Result for ", str(num_of_fold), " -fold")

    	print("accuracy= ",accuracy)



dataPath = '/home/kafein/plateRecognition/text'
imageData, targetData = labeledText(dataPath)

model = SVC(kernel='linear', probability=True)

trainAndTest(model, 4, imageData, targetData)

model.fit(imageData, targetData)

filename = 'txtClsModel.p'
pickle.dump(model, open(filename, 'wb'))
print "model saved"



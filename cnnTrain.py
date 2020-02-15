import sklearn
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers, utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Activation, Flatten, Dense, Conv2D, Lambda, Cropping2D, Dropout, LeakyReLU, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle, randint
import numpy as np
import cv2
import csv
import os
from os import walk
from os import path
import time
from sklearn.preprocessing import StandardScaler

def CNN():

	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 64, 3)))
	model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2)))
	model.add(LeakyReLU(alpha=.01))
	model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2)))
	model.add(LeakyReLU(alpha=.01))
	model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2)))
	model.add(LeakyReLU(alpha=.01))
	model.add(Dropout(0.5))
	model.add(Conv2D(64, (3, 3)))
	model.add(LeakyReLU(alpha=.01))
	model.add(Conv2D(64, (3, 3)))
	model.add(LeakyReLU(alpha=.01))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.summary()

	return model

def get_fileNames(path):
	data = []
	for root, dirs, files in os.walk(path):
		for name in files:
			if name.endswith((".jpg", ".jpeg",".png")):
				baseName=os.path.join(root,name)
				img = cv2.imread(baseName)
				img = cv2.resize(img,(256,64))
				print(baseName)
				data.append(img)
	return data

def trainAndTest():

	pathPlates='/home/kafein/plateRecognition/plates/br'
	plates=get_fileNames(pathPlates)
	print(len(plates))

	pathNotPlt='/home/kafein/plateRecognition/vehicles'
	notPlt=get_fileNames(pathNotPlt)
	print(len(notPlt))

	X = np.vstack((plates, notPlt)).astype(np.float64) 
	print("shape=",X.shape)                      
	#X_scaler = StandardScaler().fit(X)
	#scaled_X = X_scaler.transform(X)
	y = np.hstack((np.ones(len(plates)), np.zeros(len(notPlt))))

	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=rand_state)

	datagen = ImageDataGenerator()

	datagen.fit(X_train)

	valgen = ImageDataGenerator()
	valgen.fit(X_test)
	
	batch_size = 32


	model = CNN()
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

	model.compile(loss='binary_crossentropy',
				optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])


	model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
						steps_per_epoch=len(X_train)/batch_size, validation_data=valgen.flow(X_test, y_test, batch_size=batch_size),
						validation_steps=len(X_test)/batch_size, epochs=30, callbacks=[checkpoint])


"""
	#model = model.fit(X_train,y_train)
	#score = round(model.score(X_test, y_test), 4)
	#print('Test Accuracy of CNN = ', score)

	modelName = 'cnnPlt.p'
	clsModel = {}
	clsModel["model"] = model
	#clsModel["scaler"] = X_scaler
	pickle.dump(clsModel, open(modelName, 'wb'))
	print ("model saved")"""

trainAndTest()








import dbFunctions as prc
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from sklearn.preprocessing import normalize
from os.path import isfile
import numpy as np
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	print ("Provide database name to work with and optionally name and path where to save NN")
	exit()
	

if not isfile(sys.argv[1]+".hdf5"):
	print ("There is no such a hdf5 database!")
	exit()

## prepare data for CNN training

## split folds into train, tune and test sets
trainData, trainLabels, tuneData, tuneLabels, testData, testLabels = prc.split(16, sys.argv[1], [1,2,3,4,5,6,7,8],[9,10],[9,10])

## lists to np arrays
trainData=np.array(trainData)
trainLabels=np.array(trainLabels)
tuneData=np.array(tuneData)
tuneLabels=np.array(tuneLabels)

## standarize data
dbMean = np.resize(np.mean(trainData, axis = (0,2)), (1,32,1))
dbStd  = np.resize(np.std(trainData,axis = (0,2)), (1,32,1))

trainData -= dbMean
trainData /= dbStd
tuneData  -= dbMean
tuneData  /= dbStd

## classes to vector - one hot encode
trainLabels = np.array(to_categorical(trainLabels, num_classes=10))
tuneLabels = np.array(to_categorical(tuneLabels, num_classes=10))

trainData = np.expand_dims(trainData, axis=-1)
tuneData = np.expand_dims(tuneData, axis=-1)

## build model
# model = Sequential()
# model.add(Conv2D(80, (57, 6), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(60, 41, 2)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(4, 3), strides = (1,3)))
# model.add(Dropout(0.5))

# model.add(Conv2D(80, (1, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(1, 3), strides = (1,3)))

# model.add(Flatten())
# model.add(Dense(5000, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.5))
# model.add(Dense(5000, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
model = Sequential()
model.add(ZeroPadding2D(padding = (1,1), input_shape=(32, 16, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding = (1,1)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(ZeroPadding2D(padding = (1,1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))





## compile and fit network

# sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

tr,tu,ls = [],[],[]
for epoch in xrange(100):
	print "\n", epoch
	history = model.fit(trainData, trainLabels, batch_size=128, epochs=1,verbose=1)
	tr.append(history.history['acc'])
	ls.append(history.history['loss'])

	##predict on folds not used in training (currently 8,9,10 which is validation set + test set)
	tu.append(model.evaluate(tuneData, tuneLabels, batch_size=1024)[1])

#show scores
plt.plot(tr)
plt.plot(tu)
plt.title('Model accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(ls)
plt.title('Model loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

prc.predictOnDataBase(model,16, sys.argv[1], dbMean, dbStd,9,10)
scores = prc.getScores(sys.argv[1],9,10)

print (scores[1])
print (scores[0])
normscores =normalize(scores[0],norm='l1' ,axis=1)
plt.imshow(normscores)
plt.colorbar()
plt.show()

if len(sys.argv) > 2:
	model.save(sys.argv[2])



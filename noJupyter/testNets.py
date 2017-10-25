from keras.models import Sequential, load_model
import os
import dbFunctions as prc
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
import numpy as np

models = []
for x in os.listdir("models/own"):
	models.append(load_model("models/own/"+ x))

trainData, trainLabels, tuneData, tuneLabels, testData, testLabels = prc.split(16, sys.argv[1], [1,2,3,4,5,6,7,8],[9,10],[9,10])

## lists to np arrays
trainData=np.array(trainData)
trainLabels=np.array(trainLabels)
tuneData=np.array(tuneData)
tuneLabels=np.array(tuneLabels)

## standarize data
dbMean = np.resize(np.mean(trainData, axis = (0,2)), (1,64,1))
dbStd  = np.resize(np.std(trainData,axis = (0,2)), (1,64,1))

scores = []

prc.predictOnDataBase(models,16, sys.argv[1], dbMean, dbStd,9,10)
scores.append(prc.getScores(sys.argv[1],9,10))

for score in scores:

	print (score[1])
	print (score[0])
	normscores =normalize(score[0],norm='l1' ,axis=1)
	fig, ax = plt.subplots()
	ax.matshow(normscores, cmap = plt.cm.Blues)
	for i in xrange(normscores.shape[0]):
		for j in xrange(normscores.shape[1]):
			ax.text(i, j, str(int(score[0][j,i])), va='center', ha = 'center')
	# plt.colorbar()
	plt.show()

final = 0
for score in scores:
	final += score[1]

final/=len(scores)
print (final)
import librosa.display as disp

## math and charts
import numpy as np
import matplotlib.pyplot as plt


## data read/write with hdf5 files
import h5py

# splits data to train and test arrays, optional tune.
# arrays are given in form of trainData, trainLabels...
# method shall be fed with step between windows (equivalent to their width) and lists of folds that shall create 
# train, (tune), test arrays

# train and tune arrays split tracks into windows of widht = spltStep, test array preserves whole track

def split (splitStep, dataset, *data):
    
    with h5py.File(dataset +".hdf5") as f:
        
        #no tune set
        if len(data) == 2:
            trainData, trainLabels, testData, testLabels = [],[],[],[]
            
            #create train arrays
            for fold in data[0]:
                grp = f['fold' + str(fold)]
                
                for track in grp.keys():
                    
                    tempTrack = grp[track][()]
                    classNumber = grp[track].attrs['class']
                    iterator = 0

                    while iterator + splitStep <= tempTrack.shape[1]:
                        trainData.append(tempTrack[: ,iterator : iterator + splitStep,:])
                        trainLabels.append(classNumber)
                        iterator += splitStep/2
            
            #create test arrays         
            for fold in data[1]:
                grp = f['fold' + str(fold)]
                
                for track in grp.keys():
                    
                    if track.split('.')[0].split('-')[-1][:-1] == 'pitch':
                        continue
                        
                    testData.append(grp[track][()])
                    testLabels.append(grp[track].attrs['class'])
                    
            return trainData, trainLabels, testData, testLabels
                 
        # with tune set
        elif len(data) == 3:
            trainData, trainLabels, tuneData, tuneLabels, testData, testLabels = [],[],[],[],[],[]
            
            # create train arrays
            for fold in data[0]:
                grp = f['fold' + str(fold)]
                
                for track in grp.keys():
                    tempTrack = grp[track][()]
                    classNumber = grp[track].attrs['class']
                    
                    iterator = 0
                    
                        
                    while iterator + splitStep <= tempTrack.shape[1]:
                        trainData.append(tempTrack[: ,iterator : iterator + splitStep,:])
                        trainLabels.append(classNumber)
                        iterator += splitStep/2
                        
            # create tune arrays
            for fold in data[1]:
                grp = f['fold' + str(fold)]
                
                for track in grp.keys():
                    
                    if track.split('.')[0].split('-')[-1][:-1] == 'pitch':
                        continue
                    tempTrack = grp[track][()]
                    classNumber = grp[track].attrs['class']
                    
                    
                    iterator = 0
                    while iterator + splitStep <= tempTrack.shape[1]:
                        tuneData.append(tempTrack[: ,iterator : iterator + splitStep,:])
                        tuneLabels.append(classNumber)
                        iterator += splitStep/2
            
            # create test arrays
            for fold in data[2]:
                grp = f['fold' + str(fold)]
                
                for track in grp.keys():
                    
                    if track.split('.')[0].split('-')[-1][:-1] == 'pitch':
                        continue
                    
                    testData.append(grp[track][()])
                    testLabels.append(grp[track].attrs['class'])
            
            return trainData, trainLabels, tuneData, tuneLabels, testData, testLabels

        else:
            raise NameError("Wrong number of inputs")




## predicts classes on whole database or specified fold from database
def predictOnDataBase(model, splitStep, dataset, *data):
    with h5py.File(dataset +".hdf5") as f:
        ## prepare list of folds to work on
        grps = []
        if len(data) == 0:
            for grp in f:
                grps.append(f[grp])
        else:
            if type(data[0]) is list:
                for fold in data[0]:
                    grps.append(f['fold' + str(fold)])
            else:
                for fold in data:
                    grps.append(f['fold' + str(fold)])
        
        ## assign scores to every file in fold        
        for grp in grps:
            for clip in grp.keys():
                clip = grp[clip]
                if 'predictions' in clip.attrs:
                    del clip.attrs['predictions']
                
                temp = clip[()]
                trackToPredict = []
                iterator = 0

                    
                ## if clip is too short to predict even 1 window, expand it
                if temp.shape[1]<splitStep:
                    t = temp.shape[1]
                    b = np.zeros((temp.shape[0],splitStep,2))
                    b[:, :-splitStep+t,:]= temp
                    trackToPredict.append(b)
                else:
                    ## iterate through clip and split it into windows
                    while iterator + splitStep <= temp.shape[1]:
                        trackToPredict.append(temp[: ,iterator : iterator + splitStep,:])
                        iterator += splitStep
                    
                ## prepare clip as a corrent CNN input and predict probabilities
                trackToPredict= np.array(trackToPredict)
                #trackToPredict = np.expand_dims(trackToPredict, axis=3)
                predicted = model.predict(trackToPredict, batch_size=1, verbose=2)

                ## predicted is a matrix of probabilities, where each row has 10 probabilities of being
                ## specific class, each row represents one predicted window
                
                ## to predictions append list of best prediction of class for each window
                ## predicted is majority vote over all windows (but voted using probabilities, not classes)
                clip.attrs['predictions']=np.argmax(predicted,axis=1)
                clip.attrs['predicted']=predicted.sum(axis=0).argmax()
        
            

## prints score and accuracy matrix for whole dataset or for given folds
def getScores(dataset, *data):
    scores = np.zeros((10,10))
    with h5py.File(dataset + ".hdf5") as f:
        grps = []
        ## prepare folds to print scores for
        if not data:
            for grp in f:
                grps.append(f[grp])
        else:
            if type(data[0]) is list:
                for fold in data[0]:
                    grps.append(f['fold' + str(fold)])
            else:
                for fold in data:
                    grps.append(f['fold' + str(fold)])

        for grp in grps:
            for clip in grp.keys():
                clip = grp[clip]

                scores[clip.attrs['predicted'],clip.attrs['class']] += 1
                
                
            
    ## returns confusion matrix (where rows are predicted classes and columns are actuall classes)
    ## as well as accuracy 
    return scores, scores.trace()/scores.sum()
                    
        
## look for clips searching for class or/and prediction
def findClips(dataset, actual = None, predicted = None, folds = None):
    if actual == None and predicted == None:
        print ("Specify actual or predicted classes to search for")
        return None
    if actual != None:
        if type(actual) is not list:
            temp = actual
            actual = []
            actual.append(temp)
        actual = set(actual)
    if predicted != None:
        if type(predicted) is not list:
            temp = predicted
            predicted = []
            predicted.append(temp)
        predicted = set(predicted)
        
    
    
    toReturn =[]
    grps =[]
    
    with h5py.File(dataset + ".hdf5") as f:
        if folds == None:
            for grp in f:
                grps.append(f[grp])
        else:
            for fold in folds:
                grps.append('fold' + str(fold))
                    
        if actual and predicted:
            for fold in grps:
                fold = f[fold]
                for clip in fold:
                    clip = fold[clip]
                    if clip.attrs['class'] in actual and clip.attrs['predicted'] in predicted:
                        toReturn.append(clip.name)
                    
        elif actual:
            for fold in grps:
                fold = f[fold]
                for clip in fold:
                    clip = fold[clip]
                    if clip.attrs['class'] in actual:
                        toReturn.append(clip.name)
                        
        elif predicted:
            for fold in grps:
                fold = f[fold]
                for clip in fold:
                    clip = fold[clip]
                    if clip.attrs['predicted'] in predicted:
                        toReturn.append(clip.name)
    
    return toReturn

## show spectrogram of given file
def showClip(fPath, dataset):
    with h5py.File(dataset+".hdf5") as f:
        clip = f[fPath]
        print (fPath)
        print (clip[()].shape[1])
        disp.specshow(clip[()])
        plt.show()                


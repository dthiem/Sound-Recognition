import glob
import os

## sound preparation, feature extraction, preprocessing
import librosa
import librosa.display as disp
import soundfile as sf

## math and charts
import numpy as np

## data read/write with hdf5 files
import h5py

import sys
from os.path import isfile


#hop 10ms, window 40
pitchVolumes = [-2, -1, 1, 2]

def appendTracks(data,labels,paths,fp):
	X, sr = sf.read(fp)
	sound = np.array(X)
	# librosa operates on (lenght, channels) matrices, wheras soundfile gave us (channels, lenght) 
	# so we transpose
	sound = np.transpose(sound)
	# resample so every wave has same sampling rate
	sound = librosa.core.resample(sound, sr, 22050)
	# set class number
	classNumber = int(fp.split("/")[2].split("-")[1])
	# compute and set mel spectrogram
	if (sound.shape[0] == 2):

		D = [np.abs(librosa.stft(channel, hop_length=512,  win_length=1024))**2 for channel in sound]
		mel = [librosa.logamplitude(librosa.feature.melspectrogram(S=channel, sr=22050, n_mels = 60),ref_power= np.max) for channel in D]
		mel = np.array(mel)
		mel = np.mean(mel, axis = 0)
	else:
		D = np.abs(librosa.stft(sound, hop_length=512,  win_length=1024))**2
		mel = librosa.logamplitude(librosa.feature.melspectrogram(S=D, sr=22050, n_mels = 60),ref_power= np.max)
	
	delta  = librosa.feature.delta(mel)
	outData = np.array([mel,delta])
	outData = np.swapaxes(outData, 0, 2)
	outData = np.swapaxes(outData, 0, 1)
	data.append(outData)
	labels.append(classNumber)
	fp = fp.split("/")
	fp = "/".join(fp[1:])
	paths.append(fp)
	pitches = 1
	for pitch in pitchVolumes:
		if sound.shape[0] == 2:
			temp = [librosa.effects.pitch_shift(channel, 22050, pitch) for channel in sound]
			D = [np.abs(librosa.stft(channel, hop_length=512,  win_length=1024))**2 for channel in temp]
			mel = [librosa.logamplitude(librosa.feature.melspectrogram(S=channel, sr=22050, n_mels = 60),ref_power= np.max) for channel in D]
			mel = np.array(mel)
			mel = np.mean(mel, axis = 0)
		else:
			temp = librosa.effects.pitch_shift(sound, 22050, pitch)
			D = np.abs(librosa.stft(temp, hop_length=512,  win_length=1024))**2
			mel = librosa.logamplitude(librosa.feature.melspectrogram(S=D, sr=22050, n_mels = 60),ref_power= np.max)

		newFp = fp.split('.')
		newFp[0] += ('-pitch'+str(pitches)+'.')
		newFp=''.join(newFp)
		pitches += 1
		data.append(mel)
		labels.append(classNumber)
		paths.append(newFp)

from multiprocessing import Process, Lock, Pipe,Event

## DO NOT USE IF HDF5 FILE HAS BEEN ALREADY CREATED!!!

## whole thing is init method which preprocesses data and puts it into hdf5 file
## only mel scaled spectrograms and classes of sound clips are preserved


data = []

## per process method, extracts data from given fold and sends it to main process
def add(x,c):
	tempdata,templabels,tempPaths = [],[],[]
	for file in glob.glob(x):
		appendTracks(tempdata, templabels, tempPaths, file)
			
	c.send(zip(tempdata, templabels,tempPaths))
	del tempdata,templabels,tempPaths
		


if __name__ == '__main__':
	if isfile(sys.argv[1]+".hdf5"):
		print ("There is already such a hdf5 database!")
		exit()

	threadNumber = min(int(sys.argv[2]), 10)
	
	
	## create threads, assign them methods and start
	foldCounter = 1 
	while (foldCounter <= 10) :

		threads= []
		connections=[0]* threadNumber

		for x in xrange(threadNumber):
			if foldCounter >= 11 :
				del connections[x:], threads[x:]
				break
			connections[x], childPipe=Pipe()
			threads.append(Process(target=add, args=("../fold"+str(foldCounter)+"/*.wav",childPipe)))      
			foldCounter += 1   
			threads[x].start()
	
		## wait for threads to finish, get data
		for x,y in zip(threads,connections):
			tD, tL, tP = zip(*y.recv())
			data.append((tD,tL,tP))


	
## create hdf5 file and fill it with data
## format is wholeTracks/fold[NUMBER]/[trackID] and keeps info about spectrogram, as well as class number
## in attributes of dataset

with h5py.File(sys.argv[1]+".hdf5") as f:
	fold = 1
	
	for x in data:
			
		tracks, cls, paths = (x)
		for track, cl, path in zip(tracks,cls,paths):
				
			dset = f.create_dataset(path,data=track)
			dset.attrs['class']=cl       

	f.close()
	

del data





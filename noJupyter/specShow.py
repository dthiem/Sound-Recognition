import soundfile as sf
import librosa
import numpy as np
import librosa.display as disp
import matplotlib.pyplot as plt
import sys


newSrs = [44100,10000]
hops = [512, 506]
X, sr = sf.read(sys.argv[1])
sound = np.array(X)

sound = np.transpose(sound)
sounds = [librosa.core.resample(sound, sr, newSr) for newSr in newSrs]

count = 1
for sound,hop in zip(sounds,hops):
	if (sound.shape[0] == 2):

		D = [np.abs(librosa.stft(channel, hop_length=hop,  n_fft=2*hop))**2 for channel in sound]
		mel = [librosa.logamplitude(librosa.feature.melspectrogram(S=channel, sr=newSr, n_mels = 32),ref_power= np.max) for channel in D]
		mel = np.array(mel)
		mel = np.mean(mel, axis = 0)
	else:
		D = np.abs(librosa.stft(sound, hop_length=hop,  n_fft=2*hop))**2
		mel = librosa.logamplitude(librosa.feature.melspectrogram(S=D, sr=newSr, n_mels = 32),ref_power= np.max)
	plt.subplot(2,1,count)
	count+=1
	disp.specshow(mel)
plt.show()

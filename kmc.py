
import glob
import os
import matplotlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd   

######################### Helper Methods ################################


def featureExtraction(fileName):
	raw, rate = librosa.load(fileName)
	stft = np.abs(librosa.stft(raw))
	mfcc = np.mean(librosa.feature.mfcc(y=raw,sr=rate,n_mfcc=40).T, axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(raw, sr=rate).T, axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T, axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(raw), sr=rate).T, axis=0)
	return mfcc, chroma, mel, contrast, tonnetz

# Takes parent directory name, subdirectories within parent directory, and file extension as input. 
def parseAudio(parentDirectory, subDirectories, i_low, i_high, fileExtension="*.wav"):
	features, labels = np.empty((0,193)), np.empty(0)
	for subDir in subDirectories:
		print(f"Processing Sub-Directory: {subDir}")
		i = i_low
		for fn in glob.glob(os.path.join(parentDirectory, subDir, fileExtension)):
			if i == i_high:
				break
			i += 1
			print(f"\tProcessing File: {fn}")
			mfcc, chroma, mel, contrast, tonnetz = featureExtraction(fn)
			tempFeatures = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
			features = np.vstack([features, tempFeatures])
			# pop = 1, classical = 2, metal = 3, rock = 0
			global music_map
			labels = np.append(labels, music_map[subDir])
	return np.array(features), np.array(labels, dtype=np.int)

music_map = {
	"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9
}



training = "./GTZAN"
test = "./GTZAN"
subDirectories = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
# Traning Labels [1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0 0]
trainingFeatures, trainingLabels = parseAudio(training, subDirectories, 0, 5)
# Test Labels [1 1 2 2 3 3 0 0]
testFeatures, testLabels = parseAudio(test, subDirectories, 5, 10)

###################### Training Loop ######################################

model = KMeans(n_clusters=10)
model.fit(trainingFeatures)
model.predict(testFeatures)
#################### Test Results ###################################

print(model.labels_)
print(testLabels)

accuracy = 0
for i in range(model.labels_):
	if model.labels_[i] == testLabels[i]:
		accuracy += 1

print(f"Accuracy: {accuracy/len(testLabels)}")
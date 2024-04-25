import librosa
import os
import numpy as np

def featureExtraction():
# loop through all the music files in the folder music
# load the files with librosa and extract features
    rowNumber = 0
    numpyArray = np.zeros((25000, 41))  # creates a 2d array with 25000 rows for the songs and 41 columns
    for songname in os.listdir(f'./music'):
        columnNumber = 0
        y, sr = librosa.load(songname, mono=True, duration=30) # change the duration according to our test
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        lpc = librosa.lpc(y=y, order=16)  # order should be between 10 and 20 and 16 usually used
        
        # add the features to the 2D array
        for m in mfcc:
            numpyArray[rowNumber, columnNumber] = np.mean(m)
            columnNumber += 1
        for l in lpc:
            numpyArray[rowNumber, columnNumber] = np.mean(l)
            columnNumber += 1

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        rolloff = librosa.feature.spectral_rolloff( y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        numpyArray[rowNumber, columnNumber] = np.mean(zero_crossing_rate)
        columnNumber += 1
        numpyArray[rowNumber, columnNumber] = np.mean(rolloff)
        columnNumber += 1
        numpyArray[rowNumber, columnNumber] = np.mean(spectral_centroid)
        
        rowNumber += 1

    return numpyArray

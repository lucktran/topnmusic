import librosa
import os
import csv
import pandas
import torch
import numpy
import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import librosa

def extract_features_from_song(song_path):
    # first we create a row header for the csv
    header = ''
    for i in range(1, 21):
        header += f' mfcc{i}'
    for j in range(1, 17):
        header += f' lpc{j}'
    header += f' zero_crossing_rate rolloff spectral_centroid'
    header += ' genre'

    # formatting for the csv
    header = header.split()

    # create the file and add the header
    file = open('featureExtractedOneSong.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Load the new song using librosa
    y, sr = librosa.load(song_path, mono=True)

    # Extract features from the new song
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Extract 20 MFCCs
    lpc = librosa.lpc(y=y, order=16)  # Extract LPC coefficients
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    row = ''
    # Calculate means of features
    for m in mfcc:
        row += f' {numpy.mean(m)}'
    for l in lpc:
        row += f' {numpy.mean(l)}'
    zero_crossing_rate_mean = numpy.mean(zero_crossing_rate)
    rolloff_mean = numpy.mean(rolloff)
    spectral_centroid_mean = numpy.mean(spectral_centroid)

    row += f' {zero_crossing_rate_mean} {rolloff_mean} {spectral_centroid_mean}'

    file = open('featureExtractedOneSong.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(row.split())

    data = pandas.read_csv('featureExtractedOneSong.csv')

    # Concatenate all features into a single array
    features = data.values # converts csv format to numpy array

    return features

def PreProcess(features, scaler):
    # Convert features to tensor
    # new_song_tensor = torch.tensor(features, dtype=torch.float32)

    # Scale features using the same scaler used during training
    scaled_features = scaler.transform(features)

    # Convert scaled features to tensor
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # Initialize the model
    input_size = 40
    hidden_size1 = 1024  # Number of neurons for hidden layer 1 of 3
    hidden_size2 = 512
    hidden_size3 = 256
    # hidden_size4 = 128
    # hidden_size5 = 64
    output_size = 10

    # Load the saved model parameters
    loaded_model = NeuralNetwork.NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
    loaded_model.load_state_dict(torch.load('model_parameters.pth'))

    # Pass features through the trained model
    with torch.no_grad():
        output_probabilities = loaded_model(scaled_features_tensor)

    #print("output probabilities", output_probabilities)
    return output_probabilities


if __name__ == '__main__':

    # Load scaler
    scaler = joblib.load('scaler.pkl')
    
    # Example usage:
    song_path = './TestMusic/pop.wav'
    new_song_features = extract_features_from_song(song_path)

    output_probabilities = PreProcess(new_song_features, scaler)

    # Post-processing
    encoder = LabelEncoder()
    predicted_genre_index = torch.argmax(output_probabilities).item()
    encoder.fit(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    predicted_genre = encoder.inverse_transform([predicted_genre_index])[0]

    print("Predicted genre:", predicted_genre)
    print("output probabilities are:", output_probabilities)
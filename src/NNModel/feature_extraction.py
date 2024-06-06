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

def FeatureExtraction():
    """Creates a CSV file saved locally called featureExtracted.csv"""
    # first we create a row header for the csv
    header = 'filename'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for j in range(1, 18):
        header += f' lpc{j}'
    header += f' zero_crossing_rate rolloff spectral_centroid'
    header += ' genre'

    # formatting for the csv
    header = header.split()
    # create the file and add the header
    file = open('featureExtracted.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    musicGenres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    # loop through all the music files in the folder music
    # load the files with librosa and extract mfcc and lpc features
    # then add them to the csv
    for genre in musicGenres:
        print(f'{genre} started the feature extraction')
        for filename in os.listdir(f'./genres_original/{genre}'):
            songname = f'./genres_original/{genre}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30) # change the duration according to our test
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            lpc = librosa.lpc(y=y, order=16)  # order should be between 10 and 20 and 16 usually used
            row = f'{filename}'
            for m in mfcc:
                row += f' {numpy.mean(m)}'
            for l in lpc:
                row += f' {numpy.mean(l)}'
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            rolloff = librosa.feature.spectral_rolloff( y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            row += f' {numpy.mean(zero_crossing_rate)} {numpy.mean(rolloff)} {numpy.mean(spectral_centroid)}'
            row += f' {genre}'
            file = open('featureExtracted.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(row.split())

def PreProcess():
    """Preproccesses the data from the featureExtracted.csv file"""
    print('PreProcessing Started')
    data = pandas.read_csv('featureExtracted.csv')
    data.head() # Maybe don't need
    data = data.drop(['filename'], axis=1)
    data.head() # Again maybe don't need

    # Create a mapping where each genre is represented by an integer
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    # Split the data set between 90% training and 10% testing
    # Convert features to numpy array without scaling
    X = numpy.array(data.iloc[:, :-1], dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(numpy.array(data.iloc[:, :-1], dtype = float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    #Save the scaler parameters to be used in NewSongClassification.py
    joblib.dump(scaler, 'scaler.pkl')

    print("PreProcessing Finished")
    return X_train, X_test, y_train, y_test

def CreateModel(X_train, X_test, y_train, y_test):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Initialize the model
    input_size = X_train.shape[1]
    hidden_size1 = 1024 # Number of neurons for hidden layer 1 of 5
    hidden_size2 = 512
    hidden_size3 = 256
    # hidden_size4 = 128
    # hidden_size5 = 64
    output_size = 10
    model = NeuralNetwork.NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training the model
    num_epochs = 50
    batch_size = 128
    print('Training started')
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            targets = y_train_tensor[i:i+batch_size]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #added print statments to show training progression
            if (i // batch_size) % 10 == 0:
                print(f'\tBatch {i // batch_size}/{len(X_train) // batch_size}, Loss: {loss.item():.4f}')
        
    print('Training finished')

    # Save the trained model parameters
    torch.save(model.state_dict(), 'model_parameters.pth')

    # Evaluation
    with torch.no_grad():
        # Calculate accuracy
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print('Test accuracy:', accuracy)

        # Predictions
        predictions = outputs.numpy()
        print('Predicted class for the first sample:', numpy.argmax(predictions[0]))

if __name__ == '__main__':

    #FeatureExtraction()
    X_train, X_test, y_train, y_test = PreProcess()
    CreateModel(X_train, X_test, y_train, y_test)

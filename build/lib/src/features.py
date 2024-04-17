import matplotlib.pyplot as plt
import numpy as np
import librosa


def get_librosa_example_audio(key: str):
    """Gets signal and sampling rate from librosa example file

    Args:
        key (str): key for librosa example audio file. For a full list of
        keys, see:
        https://librosa.org/doc/latest/recordings.html
    """
    y, sr = librosa.load(librosa.ex(key))
    return y, sr


def get_true():
    return True


# get power spectrogram
def power_spectrum(y, sr):
    D = np.abs(librosa.stft(y))**2
    return D


# get mel spectrogram
def mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    return S


# get MFCCs
def mfcc(y, sr):
    M = librosa.feature.mfcc(y, sr=sr)
    return M


# plot mel spectrogram
def plot_mel_spectrogram(S, sr):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')


# plot MFCCs
def plot_mfcc(M):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(M, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')


# plot mel spectrogram
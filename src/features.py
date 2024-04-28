import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


def mfcc(audio_file_path: str,
         sr_target: int,
         n_fft: int,
         hop_length: int,
         n_mels: int,
         filter_name: str,
         plot: bool = False):

    y, sr = librosa.load(audio_file_path, sr=sr_target)
    win_func = librosa.filters.get_window(filter_name, n_fft)

    mfccs = librosa.feature.mfcc(y=y,
                                 sr=sr,
                                 n_fft=n_fft,
                                 hop_length=hop_length,
                                 n_mels=n_mels,
                                 window=win_func)

    if plot is True:
        S = librosa.feature.melspectrogram(y=y,
                                           sr=sr,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           n_mels=n_mels,
                                           window=win_func)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 12))
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       x_axis='time', y_axis='mel', fmax=8000,
                                       ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title='MFCC')
        plt.savefig(os.path.join("figures", "temp", "spectrogram_mfccs.png"))

    return mfccs


if __name__ == '__main__':
    audio_file_path = os.path.join('.', 'training_data', 'DEAM', '2.mp3')

    mfccs = mfcc(audio_file_path=audio_file_path,
                 sr_target=22050,
                 n_fft=5512,
                 hop_length=2205,
                 n_mels=13,
                 filter_name='hann',
                 plot=True
                 )

    print('done')

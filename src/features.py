<<<<<<< HEAD
import json
=======
>>>>>>> main
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


<<<<<<< HEAD
def get_sample(mfcc_params_set):
    # replace with luck's code
    params = load_mfcc_params(mfcc_params_set)
    audio_file_path = os.path.join('.', 'training_data', 'DEAM', '2.mp3')
    y, sr = librosa.load(audio_file_path, sr=params["sr"])
    if sr != params["sr"]:
        print("Warning: target sr is not equal to actual sr")

    return y[:params["audio_num_samples"]]


def load_mfcc_params(mfcc_params_set: str):
    json_file_path = os.path.join('.', 'src', 'mfcc_params.json')
    with open(json_file_path) as json_file:
        mfcc_param_sets = json.load(json_file)
    return mfcc_param_sets["mfcc_preprocess_config"][mfcc_params_set]


def mfcc(y: np.ndarray,
         mfcc_params_set: str,
         plot: bool = False):

    params = load_mfcc_params(mfcc_params_set)

    mfccs = librosa.feature.mfcc(y=y,
                                 sr=params["sr"],
                                 n_mfcc=params["n_mfcc"],
                                 dct_type=params["dct_type"],
                                 norm=params["norm"],
                                 lifter=params["lifter"],
                                 n_fft=params["n_fft"],
                                 hop_length=params["hop_length"],
                                 win_length=params["win_length"],
                                 window=params["window"],
                                 center=params["center"],
                                 pad_mode=params["pad_mode"],
                                 power=params["power"],
                                 n_mels=params["n_mels"],
                                 fmin=params["fmin"],
                                 fmax=params["fmax"])

    if plot is True:
        S = librosa.feature.melspectrogram(y=y,
                                           sr=params["sr"],
                                           n_fft=params["n_fft"],
                                           hop_length=params["hop_length"],
                                           win_length=params["win_length"],
                                           window=params["window"],
                                           center=params["center"],
                                           pad_mode=params["pad_mode"],
                                           power=params["power"],
                                           n_mels=params["n_mels"],
                                           fmin=params["fmin"],
                                           fmax=params["fmax"])

        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 12))
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       sr=params["sr"],
                                       hop_length=params["hop_length"],
                                       n_fft=params["n_fft"],
                                       win_length=params["win_length"],
                                       x_axis='time',
                                       y_axis='mel',
                                       fmin=params["fmin"],
                                       fmax=params["fmax"],
                                       ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title=f'Mel spectrogram, n_mels = {params["n_mels"]}')
        ax[0].label_outer()
        img = librosa.display.specshow(mfccs,
                                       sr=params["sr"],
                                       hop_length=params["hop_length"],
                                       n_fft=params["n_fft"],
                                       win_length=params["win_length"],
                                       x_axis='time',
                                       fmin=params["fmin"],
                                       fmax=params["fmax"],
                                       ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title=f'MFCC, n_mfcc = {params["n_mfcc"]}')
        ax[1].set(ylabel="MFCC Number")
=======
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
>>>>>>> main
        plt.savefig(os.path.join("figures", "temp", "spectrogram_mfccs.png"))

    return mfccs


if __name__ == '__main__':
<<<<<<< HEAD
    mfcc_params_set = "fmax_all_music"
    y = get_sample(mfcc_params_set)
    M = mfcc(y, mfcc_params_set, plot=True)
=======
    audio_file_path = os.path.join('.', 'training_data', 'DEAM', '2.mp3')

    mfccs = mfcc(audio_file_path=audio_file_path,
                 sr_target=22050,
                 n_fft=5512,
                 hop_length=2205,
                 n_mels=13,
                 filter_name='hann',
                 plot=True
                 )
>>>>>>> main

    print('done')

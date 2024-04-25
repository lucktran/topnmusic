import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


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

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 12))
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       x_axis='time', y_axis='mel',
                                       fmax=params["fmax"], ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title='MFCC')
        plt.savefig(os.path.join("figures", "temp", "spectrogram_mfccs.png"))

    return mfccs


if __name__ == '__main__':
    mfcc_params_set = "fmax_most_music"
    y = get_sample(mfcc_params_set)
    M = mfcc(y, mfcc_params_set, plot=True)

    print('done')

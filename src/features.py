import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import shutil
from sklearn.model_selection import train_test_split


def get_audio_data(audio_set: str) -> pd.DataFrame:
    audio_folder_path = os.path.join(".", "data", "raw", audio_set)
    labels_path = os.path.join(audio_folder_path, "labeled_tracks.csv")
    audio_data = pd.read_csv(labels_path)
    return audio_data


def load_preprocess_params(preprocess_param_set: str) -> dict:
    json_file_path = os.path.join('.', 'src', 'preprocess_params.json')
    with open(json_file_path) as json_file:
        preprocess_params_f = json.load(json_file)
    return preprocess_params_f["preprocess_param_sets"][preprocess_param_set]


def load_audio_file(audio_file_path: str,
                    sr_target: int
                    ) -> np.ndarray:
    raw_folder = os.path.join(".", "data", "raw")
    full_audio_path = os.path.join(raw_folder, audio_file_path)
    y, sr = librosa.load(full_audio_path, sr=sr_target)
    if sr != sr_target:
        print(f"Warning: for file at path {audio_file_path}, sr target \
              {sr_target} is not equal to actual sr {sr}")
    return y


def get_non_silent_segments(y: np.ndarray,
                            frame_length: int,
                            hop_length: int,
                            audio_num_samples: int
                            ) -> np.ndarray:
    # trim end silence - with fine frame length and hop length
    yt, index = librosa.effects.trim(y,
                                     frame_length=frame_length,
                                     hop_length=hop_length)

    # trim middle silence - with audio sample length, non-overlapping
    intervals = librosa.effects.split(yt,
                                      frame_length=audio_num_samples,
                                      hop_length=audio_num_samples)

    # put non-silent audio intervals together
    y_ns = yt[intervals[0][0]:intervals[0][1]]
    for interval in intervals[1:]:
        y_ns = np.append(y_ns, yt[interval[0]:interval[1]])

    # truncate so that it divides evenly into the segments
    end_cutoff = y_ns.size % audio_num_samples
    y_trun = y_ns[:-end_cutoff]

    # create segments
    segs = np.lib.stride_tricks.sliding_window_view(y_trun, audio_num_samples)[::audio_num_samples]

    return segs


def mfcc(y: np.ndarray,
         params: dict,
         plot: bool = False
         ) -> np.ndarray:

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
        plt.savefig(os.path.join("figures", "temp", "spectrogram_mfccs.png"))

    return mfccs


def zero_crossing_rate(y: np.ndarray,
                       params: dict
                       ) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(y,
                                             frame_length=params["win_length"],
                                             hop_length=params["hop_length"],
                                             center=params["center"])
    return zcr


def spectral_rolloff(y: np.ndarray,
                     params: dict
                     ) -> np.ndarray:
    s_r = librosa.feature.spectral_rolloff(y=y,
                                           sr=params["sr"],
                                           n_fft=params["n_fft"],
                                           hop_length=params["hop_length"],
                                           window=params["window"],
                                           center=params["center"],
                                           pad_mode=params["pad_mode"])
    return s_r


def spectral_centroid(y: np.ndarray,
                      params: dict
                      ) -> np.ndarray:
    s_c = librosa.feature.spectral_centroid(y=y,
                                            sr=params["sr"],
                                            n_fft=params["n_fft"],
                                            hop_length=params["hop_length"],
                                            window=params["window"],
                                            center=params["center"],
                                            pad_mode=params["pad_mode"])
    return s_c


def prepare_folder(audio_set: str, param_set: str) -> str:
    # prepare samples folder
    samp_folder_name = f"samples_audio_{audio_set}_params_{param_set}"
    samp_folder_path = os.path.join("data", "samples", samp_folder_name)
    if os.path.exists(samp_folder_path):
        shutil.rmtree(samp_folder_path)
    os.mkdir(samp_folder_path)
    return samp_folder_path


def create_samples(segs: np.ndarray,
                   params: dict,
                   df_labels: pd.DataFrame,
                   track_id: int,
                   label: str,
                   samp_folder_path: str):
    # get samples
    for i, seg in enumerate(segs):
        M = mfcc(seg, params)  # n_mfcc x num_windows
        zcr = zero_crossing_rate(seg, params)  # 1 x num_windows
        s_r = spectral_rolloff(seg, params)  # 1 x num_windows
        s_c = spectral_centroid(seg, params)  # 1 x num_windows

        # package them into one np.ndarray
        sample = np.concatenate((M, zcr, s_r, s_c)).T

        # save features
        samp_file_name = f"sample_{track_id}_{i}.pkl"
        file_path = os.path.join(samp_folder_path, samp_file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(sample, f)

        # remember the label and original track of this sample
        df_labels.loc[len(df_labels.index)] = [samp_file_name, label, track_id]


def save_labels(samp_folder_path: str, df_labels: pd.DataFrame):
    csv_path = os.path.join(samp_folder_path, "labels.csv")
    df_labels.to_csv(csv_path)


def divide_samples(df_labels: pd.DataFrame,
                   data_split: list
                   ) -> tuple:
    # plot distribution
    # df_labels['label'].value_counts().plot(kind='pie', autopct='%1.0f%%')

    # split off training set
    X = df_labels['file_name']
    y = df_labels['label']

    X_train, X_non_train, y_train, y_non_train = train_test_split(
        X, y, train_size=data_split[0], random_state=42
    )
    # y_train.value_counts().plot(kind='pie', autopct='%1.0f%%')

    # split off test, validation sets
    rel_test_pct = data_split[1] / (1 - data_split[0])
    X_test, X_val, y_test, y_val = train_test_split(
        X_non_train, y_non_train, train_size=rel_test_pct, random_state=42
    )

    return X_train, X_test, X_val, y_train, y_test, y_val


def process_audio_files(audio_set: str,
                        preprocess_param_set: str):
    # load preprocess parameter set
    params = load_preprocess_params(preprocess_param_set)

    # get labels of original audio
    audio_data = get_audio_data(audio_set)

    # initialize df to save the labels of created samples
    df_labels = pd.DataFrame(columns=["file_name", "label", "track_id"])

    # prepare folder to hold samples
    samp_folder_path = prepare_folder(audio_set, preprocess_param_set)

    for track_id in audio_data["track_id"].values:
        try:
            row = audio_data.loc[audio_data['track_id'] == track_id]
            audio_file_path = row["file_path"].values[0]
            label = row["genre_top"].values[0]

            # load song
            y = load_audio_file(audio_file_path, params["sr"])

            # drop dead space and split into segments
            segs = get_non_silent_segments(y,
                                           params["win_length"],
                                           params["hop_length"],
                                           params["audio_num_samples"])

            # extract and save features
            create_samples(segs, params,
                           df_labels,
                           track_id,
                           label,
                           samp_folder_path)
        except:
            with open('problem_audio.txt', 'a') as f:
                f.write(f"{track_id}\n")

    # save lables
    save_labels(samp_folder_path, df_labels)

    # divide samples into train, test, validation sets
    X_train, X_test, X_val, y_train, y_test, y_val = divide_samples(
        samp_folder_path, df_labels)

    # get normalization data from training set

    # balance sets through upsampling


# def observe_feat_distributions(samp_folder_path: str,
#                                df_labels: pd.DataFrame):
#     for file_name in df_labels["file_name"]:
#         file_path = os.path.join(samp_folder_path, file_name)
#         sample = pickle.load(file_path)


def get_max_min(samp_folder_path: str, df_labels: pd.DataFrame):
    num_feats = 8
    maxs = np.ones((num_feats)) * -np.inf
    mins = np.ones((num_feats)) * np.inf
    for file_name in df_labels["file_name"]:
        file_path = os.path.join(samp_folder_path, file_name)
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
        sample_maxs = np.amax(sample, axis=0)
        sample_mins = np.amin(sample, axis=0)

        maxs = np.where(sample_maxs > maxs, sample_maxs, maxs)
        mins = np.where(sample_mins < mins, sample_mins, mins)

    return maxs, mins


if __name__ == '__main__':
    process_audio_files(audio_set="fma_medium",
                        preprocess_param_set="fmax_all_music")

    # preprocess_param_set = "fmax_all_music"
    # params = load_preprocess_params(preprocess_param_set)
    # samp_folder_name = "samples_audio_fma_medium_params_fmax_all_music"
    # samp_folder_path = os.path.join(".", "data", "samples", samp_folder_name)
    # df_labels_path = os.path.join(samp_folder_path, "labels.csv")
    # df_labels = pd.read_csv(df_labels_path)

    # X_train, X_test, X_val, y_train, y_test, y_val = divide_samples(df_labels, params["data_split"])
    # get_max_min(samp_folder_path, df_labels)

    print('done')

import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


def load_preprocess_params(preprocess_param_set: str
                     ) -> dict:
    json_file_path = os.path.join('.', 'src', 'preprocess_params.json')
    with open(json_file_path) as json_file:
        preprocess_params_f = json.load(json_file)
    return preprocess_params_f["preprocess_param_sets"][preprocess_param_set]


def load_audio_file(audio_file_path: str,
                    sr_target: int
                    ) -> np.ndarray:
    y, sr = librosa.load(audio_file_path, sr=sr_target)
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


def get_seg_features(segs: np.ndarray,
                     params: dict,
                     track_id: int,
                     label: str):
    # path to features, path to csv of name + label + track

    for i, seg in enumerate(segs):
        # get features
        M = mfcc(seg, params)  # n_mfcc x num_windows
        zcr = zero_crossing_rate(seg, params)  # 1 x num_windows
        s_r = spectral_rolloff(seg, params)  # 1 x num_windows
        s_c = spectral_centroid(seg, params)  # 1 x num_windows

        # package them into one np.ndarray
        features = np.concatenate((M, zcr, s_r, s_c))

        # save features



def preprocess_one_audio_file(audio_file_path: str,
                              preprocess_param_set):
    # load preprocess parameter set
    params = load_preprocess_params(preprocess_param_set)

    # load song
    y = load_audio_file(audio_file_path,
                        params["sr"])

    # drop dead space and split into segments
    segs = get_non_silent_segments(y,
                                   params["win_length"],
                                   params["hop_length"],
                                   params["audio_num_samples"])

    # extract and save features
    get_seg_features(segs, params, 'rock')

    print('done here')


if __name__ == '__main__':
    mfcc_params_set = "fmax_all_music"
    y = preprocess_one_audio_file(audio_file_path=os.path.join('.', 'training_data', 'DEAM', '2.mp3'),
                                  preprocess_param_set="fmax_all_music")

    print('done')

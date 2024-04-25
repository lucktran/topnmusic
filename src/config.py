mfcc_preprocess_config = {
    "fmax_most_music": {
        "sr": 8_000,  # Hz
        "n_mfcc": 5,
        "dct_type": 2,
        "norm": "ortho",
        "lifter": 0,  # 0 means no lifter
        "n_fft": 256,
        "hop_length": 102,
        "win_length": 256,
        "window": "hann",
        "center": False,
        "pad_mode": True,
        "power": 2,
        "n_mels": 26,
        "fmin": 20,  # Hz
        "fmax": 4_000,  # Hz
    },

    "fmax_all_music": {
        "sr": 10_000,  # Hz
        "n_mfcc": 5,
        "dct_type": 2,
        "norm": "ortho",
        "lifter": 0,  # 0 means no lifter
        "n_fft": 256,
        "hop_length": 102,
        "win_length": 256,
        "window": "hann",
        "center": False,
        "pad_mode": True,
        "power": 2,
        "n_mels": 26,
        "fmin": 20,  # Hz
        "fmax": 5_000,  # Hz
    },

    "fmax_more_than_music": {
        "sr": 16_000,  # Hz
        "n_mfcc": 5,
        "dct_type": 2,
        "norm": "ortho",
        "lifter": 0,  # 0 means no lifter
        "n_fft": 512,
        "hop_length": 205,
        "win_length": 512,
        "window": "hann",
        "center": False,
        "pad_mode": True,
        "power": 2,
        "n_mels": 26,
        "fmin": 20,  # Hz
        "fmax": 8_000,  # Hz
    }
}

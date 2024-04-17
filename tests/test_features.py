import unittest

import src.features


class TestMFCCFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # get sample audio
        y, sr = src.features.get_librosa_example_audio('brahms')
        cls._y = y
        cls._sr = sr

    def test_mel_spectrogram(self):
        src.features.mel_spectrogram(self._y, self._sr)

    # def test_plot_mel_spectrogram(self):
    #     S = src.features.mel_spectrogram(self._y, self._sr)
    #     src.features.plot_mel_spectrogram(S, self._sr)

    # def test_mfcc(self):
    #     M = src.features.mfcc(self._y, self._sr)


if __name__ == '__main__':
    unittest.main()

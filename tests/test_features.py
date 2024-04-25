import os
import unittest

import src.features


class TestMFCCFeatures(unittest.TestCase):

    def test_mfcc_shape_0(self):
        audio_file_path = os.path.join('.', 'training_data', 'DEAM', '2.mp3')
        n_mels = 13
        mfccs = src.features.mfcc(audio_file_path=audio_file_path,
                                  sr_target=22050,
                                  n_fft=5512,
                                  hop_length=2205,
                                  n_mels=n_mels,
                                  filter_name='hann',
                                  plot=False
                                  )
        self.assertEqual(n_mels, mfccs.shape[0])


if __name__ == '__main__':
    unittest.main()

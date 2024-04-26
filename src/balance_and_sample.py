import os
import pandas as pd
import librosa
import soundfile as sf
from sklearn.utils import resample


def load_labeled_data(csv_path):
    """
    Load the labeled data from a CSV file.
    """
    return pd.read_csv(csv_path, header=None, names=['track_id', 'file_path', 'genre_top'])


def balance_genres(metadata, genre_column='genre_top'):
    """
    Balances the genres in the metadata DataFrame by oversampling to the level of the most populous genre.
    """
    if genre_column not in metadata.columns:
        raise ValueError(f"{genre_column} column is not in the DataFrame")

    # Calculate the max genre
    max_genre_count = metadata[genre_column].value_counts().max()

    balanced_metadata = pd.DataFrame()
    for genre in metadata[genre_column].unique():
        genre_data = metadata[metadata[genre_column] == genre]
        # Oversample
        resampled_genre_data = resample(genre_data,
                                        replace=True,
                                        n_samples=max_genre_count,
                                        random_state=123)
        balanced_metadata = pd.concat([balanced_metadata, resampled_genre_data])

    return balanced_metadata


def segment_and_save_audio(audio_directory, output_directory, balanced_metadata, segment_length=5, target_sr=22050):
    """
    Load audio files, resample them, segment into non-overlapping sections, and save the segments.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files_processed = 0
    files_failed = []

    for index, row in balanced_metadata.iterrows():
        file_path = os.path.join(audio_directory, row['file_path'])
        if file_path.endswith('.mp3'):
            try:
                audio, sr = librosa.load(file_path, sr=target_sr)
                samples_per_segment = segment_length * target_sr
                num_segments = int(len(audio) / samples_per_segment)
                for i in range(num_segments):
                    start_sample = i * samples_per_segment
                    end_sample = start_sample + samples_per_segment
                    segment = audio[start_sample:end_sample]
                    segment_filename = f"{os.path.splitext(row['file_path'])[0]}_segment_{i}.mp3"
                    segment_path = os.path.join(output_directory, segment_filename)
                    sf.write(segment_path, segment, target_sr)
                    print(f"Segment saved: {segment_path}")
                files_processed += 1
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                files_failed.append(file_path)

    print(f"Total files processed: {files_processed}")
    print(f"Files failed: {files_failed}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    audio_dir = os.path.join(data_dir, 'raw', 'fma_medium')
    tracks_csv = os.path.join(data_dir, 'raw', 'tracks.csv')
    output_csv = os.path.join(data_dir, 'processed', 'labeled_tracks.csv')

    labeled_data = load_labeled_data(tracks_csv)
    balanced_metadata = balance_genres(labeled_data)
    output_dir = os.path.join(data_dir, 'processed', 'audio_segments')
    segment_and_save_audio(audio_dir, output_dir, balanced_metadata)
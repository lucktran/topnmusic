import os
import pandas as pd


def label_tracks(audio_directory, tracks_csv_path, output_csv_path):
    """
    Label tracks with genre and track ID, and save to a new CSV file.
    """

    # Metadata from the tracks.csv file
    tracks_df = pd.read_csv(tracks_csv_path, index_col=0, skiprows=1)

    # Filter for genre
    if 'genre_top' in tracks_df.columns:
        tracks_df = tracks_df[['genre_top']].dropna()
    else:
        raise ValueError("Does not contain 'genre_top' column.")

    # Create list to store track info
    track_info = []

    # Walk through the directory containing the audio files
    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.mp3'):
                track_id = int(os.path.splitext(file)[0])
                file_path = os.path.join(root, file)
                track_info.append({'track_id': track_id, 'file_path': file_path})

    # Convert to DF
    audio_df = pd.DataFrame(track_info)

    # Include audio file path
    labeled_tracks_df = pd.merge(audio_df, tracks_df, how='inner', left_on='track_id', right_index=True)

    # Create new CSV
    labeled_tracks_df.to_csv(output_csv_path, index=False)

    print(f"Data saved to {output_csv_path}")


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    audio_dir = os.path.join(data_dir, 'raw', 'fma_medium')
    tracks_csv = os.path.join(data_dir, 'raw', 'tracks.csv')
    output_csv = os.path.join(data_dir, 'processed', 'labeled_tracks.csv')

    label_tracks(audio_dir, tracks_csv, output_csv)

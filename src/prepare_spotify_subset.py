from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "data" / "spotify_tracks_selected.csv"
RAW_DATA_URL = "https://raw.githubusercontent.com/sushmaakoju/spotify-tracks-data-analysis/main/SpotifyFeatures.csv"
SELECTED_GENRES = ["Pop", "Rock", "Hip-Hop", "Electronic", "Classical"]
KEEP_COLUMNS = [
    "genre",
    "artist_name",
    "track_name",
    "track_id",
    "popularity",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


def main() -> None:
    dataframe = pd.read_csv(RAW_DATA_URL)
    dataframe["genre"] = dataframe["genre"].replace({"Children’s Music": "Children's Music"})
    dataframe = dataframe[dataframe["genre"].isin(SELECTED_GENRES)][KEEP_COLUMNS].dropna().copy()
    dataframe = dataframe.groupby("genre", group_keys=False).sample(n=1500, random_state=42).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(dataframe["genre"].value_counts().to_string())


if __name__ == "__main__":
    main()

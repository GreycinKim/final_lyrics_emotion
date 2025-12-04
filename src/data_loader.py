# src/data_loader.py

import pandas as pd
from typing import List, Dict

def fix_mojibake(text: str) -> str:
    """
    Try to fix encoding issues like 'AxÃ©' -> 'Axé'.
    If it fails, just return the original text.
    """
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text

def segment_lyrics(lyrics: str) -> List[str]:
    """
    Split lyrics into non-empty lines.
    Each line becomes one 'segment' for emotion prediction.
    """
    if not isinstance(lyrics, str):
        return []

    # Normalize line endings
    text = lyrics.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines

def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a text file with columns like:
      artist_name, song_name, genres, language, lyrics, artist_popularity, new_artist_popularity

    We let pandas auto-detect the separator (comma, tab, etc.)
    to avoid parser errors.
    """
    # Auto-detect separator using the Python engine
    df = pd.read_csv(csv_path, sep=None, engine="python")
    print("Loaded", len(df), "rows from", csv_path)
    print("Columns in CSV:", df.columns.tolist())

    songs = []

    for _, row in df.iterrows():
        # --- Artist ---
        if "artist" in df.columns:
            artist = row.get("artist", "Unknown Artist")
        else:
            artist = row.get("artist_name", "Unknown Artist")

        # --- Song name ---
        if "song" in df.columns:
            song = row.get("song", "Unknown Song")
        else:
            song = row.get("song_name", "Unknown Song")

        # --- Genre ---
        if "genre" in df.columns:
            genre = row.get("genre", "Unknown Genre")
        else:
            # your old file used "genres"
            genre = row.get("genres", "Unknown Genre")

        # --- Album ---
        # you don't have an album column yet, so this will just say "Unknown Album"
        if "album" in df.columns:
            album = row.get("album", "Unknown Album")
        elif "album_name" in df.columns:
            album = row.get("album_name", "Unknown Album")
        else:
            album = "Unknown Album"

        # --- Lyrics ---
        lyrics = row.get("lyrics", "")

        # Fix mojibake in genres and lyrics
        artist = str(artist)
        song = str(song)
        genre = fix_mojibake(str(genre))
        lyrics = fix_mojibake(str(lyrics))

        segments = segment_lyrics(lyrics)

        song_info = {
            "artist": artist,
            "album": album,
            "song": song,
            "genre": genre,
            "segments": segments,
        }
        songs.append(song_info)

    return songs

if __name__ == "__main__":
    from .config import SONG_CSV_PATH
    songs = load_songs(SONG_CSV_PATH)
    print("First song:", songs[0]["artist"], "-", songs[0]["song"])
    print("Genre:", songs[0]["genre"])
    print("First 5 segments:")
    for line in songs[0]["segments"][:5]:
        print("  •", line)

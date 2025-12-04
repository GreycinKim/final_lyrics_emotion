# src/visualization.py

import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

from .config import EMOTIONS, OUTPUT_TIMELINES, OUTPUT_ALBUMS

def plot_song_timeline(song_info: Dict, segment_results: List[Dict]):
    """
    Draw a scatter plot:
    x-axis = segment index
    y-axis = emotion (0..len(EMOTIONS)-1)
    """
    Path(OUTPUT_TIMELINES).mkdir(parents=True, exist_ok=True)

    x = [seg["segment_index"] for seg in segment_results]
    labels = [seg["label"] for seg in segment_results]

    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    y = [emotion_to_idx[label] for label in labels]

    plt.figure(figsize=(10, 3))
    plt.scatter(x, y)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Segment")
    plt.title(f"Emotion Timeline: {song_info['artist']} - {song_info['song']}")
    plt.tight_layout()

    safe_name = f"{song_info['artist']}_{song_info['song']}".replace(" ", "_")
    out_path = os.path.join(OUTPUT_TIMELINES, f"{safe_name}_timeline.png")
    plt.savefig(out_path)
    plt.close()

def plot_album_overview(album_name: str, artist: str, song_summaries: List[Dict]):
    """
    For an artist + album, show each song and its main emotion.
    song_summaries: list of dicts:
      {
        "song": song_name,
        "main_emotion": "happy",
        "genre": "Pop"
      }
    """
    Path(OUTPUT_ALBUMS).mkdir(parents=True, exist_ok=True)

    songs = [s["song"] for s in song_summaries]
    main_emotions = [s["main_emotion"] for s in song_summaries]

    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    y = [emotion_to_idx[e] for e in main_emotions]
    x = list(range(len(songs)))

    plt.figure(figsize=(max(8, len(songs) * 0.7), 4))
    plt.scatter(x, y)
    plt.xticks(x, songs, rotation=45, ha="right")
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Song")
    plt.ylabel("Main Emotion")
    plt.title(f"Album Emotion Overview: {artist} - {album_name}")
    plt.tight_layout()

    safe_album = album_name.replace(" ", "_")
    safe_artist = artist.replace(" ", "_")
    out_path = os.path.join(OUTPUT_ALBUMS, f"{safe_artist}_{safe_album}_album.png")
    plt.savefig(out_path)
    plt.close()

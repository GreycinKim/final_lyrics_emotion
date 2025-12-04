# src/run_pipeline.py

import os
import re
from pathlib import Path
from collections import defaultdict, Counter

from .config import OUTPUT_SUMMARIES, SONG_CSV_PATH
from .data_loader import load_songs
from .emotion_predict import load_emotion_model, predict_segment_emotions
from .visualization import plot_song_timeline, plot_album_overview
from .summary_llm import make_summary


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so",
    "to", "of", "in", "on", "at", "for", "from",
    "is", "are", "am", "was", "were", "be", "been",
    "i", "you", "we", "they", "he", "she", "it",
    "me", "my", "your", "our", "their",
    "im", "youre", "dont", "cant", "aint",
}


def simple_tokenize(text: str):
    """Very simple word tokenizer for counting top words."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
    cleaned = []
    for t in tokens:
        t = t.strip("'")
        if len(t) <= 2:
            continue
        if t in STOPWORDS:
            continue
        cleaned.append(t)
    return cleaned


def get_top_words_by_emotion(seg_results, top_n=5):
    """
    Build a dict: emotion -> list of top words (no scores),
    based on simple word counts in segments of that emotion.
    """
    # emotion -> Counter(words)
    emo_counts = defaultdict(Counter)

    for seg in seg_results:
        label = seg["label"]
        tokens = simple_tokenize(seg["text"])
        emo_counts[label].update(tokens)

    top_words = {}
    for emotion, counter in emo_counts.items():
        most_common = counter.most_common(top_n)
        top_words[emotion] = [w for w, _ in most_common]

    return top_words


def find_main_emotion(segment_results):
    labels = [seg["label"] for seg in segment_results]
    c = Counter(labels)
    return c.most_common(1)[0][0]


def run():
    Path(OUTPUT_SUMMARIES).mkdir(parents=True, exist_ok=True)

    # 1. Load songs and model
    songs = load_songs(SONG_CSV_PATH)
    tokenizer, model, device = load_emotion_model()

    # Keep track of album info
    # key: (artist, album) -> list of song summaries
    album_map = defaultdict(list)

    for song in songs:
        print("Processing:", song["artist"], "-", song["song"])
        segments = song["segments"]

        if not segments:
            print("  No lyrics, skipping.")
            continue

        # 2. Predict emotions for each segment
        seg_results = predict_segment_emotions(segments, tokenizer, model, device)

        # 3. Plot timeline
        plot_song_timeline(song, seg_results)

        # 4. Build stats for summary
        # Emotion counts
        emotion_counts = {}
        for seg in seg_results:
            label = seg["label"]
            emotion_counts[label] = emotion_counts.get(label, 0) + 1

        # Emotion transitions
        transitions = []
        prev = None
        for seg in seg_results:
            cur = seg["label"]
            if prev is not None and cur != prev:
                transitions.append((prev, cur, seg["segment_index"]))
            prev = cur

        # Top words per emotion (simple frequency)
        top_words = get_top_words_by_emotion(seg_results, top_n=5)

        # Song title for the LLM
        song_id = f"{song['artist']} - {song['song']}"

        # 5. Call ChatGPT to make the summary text
        summary = make_summary(
            song_id,
            seg_results,
            emotion_counts,
            transitions,
            top_words,
        )

        # 6. Save summary to a text file
        safe_name = f"{song['artist']}_{song['song']}".replace(" ", "_")
        summary_path = os.path.join(OUTPUT_SUMMARIES, f"{safe_name}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print("  Summary saved to:", summary_path)

        # 7. Save album info (for album overview plot)
        main_emotion = find_main_emotion(seg_results)
        album_key = (song["artist"], song["album"])
        album_map[album_key].append({
            "song": song["song"],
            "main_emotion": main_emotion,
            "genre": song["genre"],
        })

    # 8. For each album, draw a simple album overview plot
    for (artist, album), song_list in album_map.items():
        if len(song_list) > 1:
            print("Building album overview for:", artist, "-", album)
            plot_album_overview(album, artist, song_list)


if __name__ == "__main__":
    run()

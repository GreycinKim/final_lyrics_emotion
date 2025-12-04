# src/summary_llm.py

from openai import OpenAI

client = OpenAI(api_key="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")

def make_summary(song_title, segments, emotion_counts, transitions, top_words):
    """
    Creates a 2-paragraph natural language summary using ChatGPT.
    """
    # Build emotion distribution text
    total = sum(emotion_counts.values())
    dist_text = ", ".join([f"{emotion}: {count}/{total}" for emotion, count in emotion_counts.items()])

    # Transition text
    transitions_text = (
        "; ".join([f"{a} → {b} at segment {i}" for (a, b, i) in transitions])
        if transitions else "No major emotion changes detected."
    )

    # Top words text
    top_words_text = "\n".join([
        f"{emotion}: {', '.join(words)}"
        for emotion, words in top_words.items()
    ])

    prompt = f"""
You analyze song lyrics using emotion statistics.

Song: {song_title}

Emotion distribution:
{dist_text}

Emotion transitions:
{transitions_text}

Top emotion-related words:
{top_words_text}

Write **two short, simple paragraphs** describing:
1) The overall emotional theme of the song and how it changes.
2) Which emotions dominate the beginning, middle, and end, using the provided words.

Do NOT quote lyrics. Use only the emotions and words above.
Make it sound natural and easy to understand.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.6,
    )

    return resp.choices[0].message.content.strip()

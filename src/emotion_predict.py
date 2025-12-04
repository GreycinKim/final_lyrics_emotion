# src/emotion_predict.py

import os
from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import EMOTIONS, MODEL_NAME, MODEL_SAVE_DIR, MAX_LENGTH


def load_emotion_model():
    """
    Load the fine-tuned model if it exists, otherwise fall back to the base model.
    """
    if os.path.isdir(MODEL_SAVE_DIR):
        print(f"Loading fine-tuned model from: {MODEL_SAVE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    else:
        print("WARNING: Fine-tuned model not found. Falling back to base model.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_segment_emotions(
    segments: List[str],
    tokenizer,
    model,
    device,
) -> List[Dict]:
    results = []
    for idx, text in enumerate(segments, start=1):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        label_idx = int(probs.argmax())
        label = EMOTIONS[label_idx]

        results.append({
            "segment_index": idx,
            "text": text,
            "probs": probs.tolist(),
            "label": label,
        })
    return results

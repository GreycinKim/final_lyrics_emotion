# src/train_model.py

import os
import pandas as pd
import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .config import (
    EMOTIONS,
    MODEL_NAME,
    MODEL_SAVE_DIR,
    TRAIN_SEGMENTS_PATH,
    MAX_LENGTH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)


class EmotionDataset(Dataset):
    """
    Simple PyTorch dataset for (text, label) emotion data.
    """

    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label_str = str(row["label"]).strip().lower()

        # Map label string -> integer id (0..len(EMOTIONS)-1)
        label_id = EMOTIONS.index(label_str)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_id, dtype=torch.long)
        return item


def main():
    # 1. Load and clean training data
    if not os.path.exists(TRAIN_SEGMENTS_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAIN_SEGMENTS_PATH}")

    df = pd.read_csv(
        TRAIN_SEGMENTS_PATH,
        sep=None,  # auto-detects comma, tab, etc.
        engine="python",  # more flexible parser
        on_bad_lines="skip",  # skip any malformed lines instead of crashing
    )
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("train_segments.csv must have columns: 'text' and 'label'")

    # Normalize labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only rows whose label is in EMOTIONS
    df = df[df["label"].isin(EMOTIONS)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows after filtering by EMOTIONS. Check your labels.")

    print("Label counts:\n", df["label"].value_counts())

    # Shuffle and split into train/val
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    print("Train size:", len(df_train))
    print("Val size:", len(df_val))

    # 2. Load tokenizer and base emotion model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        # Fallback pad token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.cls_token

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Ensure model knows pad token id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 3. Create datasets
    train_ds = EmotionDataset(df_train, tokenizer)
    val_ds = EmotionDataset(df_val, tokenizer)

    # 4. Training arguments (simple, compatible with older transformers)
    args = TrainingArguments(
        output_dir="models/checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    # 6. Train
    trainer.train()

    # 7. Save fine-tuned model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("Fine-tuned model saved to:", MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()

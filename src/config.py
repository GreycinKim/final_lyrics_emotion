# src/config.py
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Start from a pre-trained EMOTION model (not plain BERT)
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

# Where your fine-tuned model will be saved
MODEL_SAVE_DIR = "models/emotion_model"

# Data
SONG_CSV_PATH = "data/songs.csv"
TRAIN_SEGMENTS_PATH = "data/train_segments.csv"

# Training hyperparameters
MAX_LENGTH = 64
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# Outputs
OUTPUT_TIMELINES = "outputs/timelines"
OUTPUT_SUMMARIES = "outputs/summaries"
OUTPUT_ALBUMS = "outputs/albums"

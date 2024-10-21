import os, torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_LOADER_BATCH_SIZE = 128
TRAIN_EPOCHS = 100
TRAIN_LR = 0.01
TRAIN_LR_SCHEDULER_STEP_SIZE = 10
TRAIN_LR_SCHEDULER_GAMMA = 0.1

STOPWORDS_PATH = ["stopwords/hit_stopwords.txt", "stopwords/special_characters_stopwords.txt"]
TRAIN_DATA_PATH = "file/train.ndjson"
VALIDATION_DATA_PATH = "file/validation.ndjson"
TEST_DATA_PATH = "file/test.ndjson"

SEQUENCE_LENGTH = 50
WORD2VEC_VECTOR_SIZE = 100
WORD2VEC_EPOCHS = 10
WORD2VEC_MIN_COUNT = 2

MODEL_HIDDEN_SIZE = 1024
MODEL_PTH_FILENAME = f"pt/sentiment_analysis_{TRAIN_EPOCHS}_{TRAIN_LR}.pt"

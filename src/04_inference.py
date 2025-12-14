# 04_inference.py
import os
import torch
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, TensorDataset

from config import *
from utils import setup_logger
from model import MultiLayerLSTM

# ==================================================
# LOGGER
# ==================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
logger = setup_logger(LOG_FILE)
logger.info("\n\n==== INFERENCE START ====")

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ==================================================
# LOAD INFERENCE DATA (NO LABEL REQUIRED)
# ==================================================
inference_csv = os.path.join(DATA_DIR, "inference.csv")
df = pd.read_csv(inference_csv).dropna(subset=["text"])
logger.info(f"Loaded {len(df)} inference samples")

texts = df["text"].astype(str).tolist()
tokenized = [t.split()[:SEQ_LEN] for t in texts]

# ==================================================
# LOAD WORD2VEC
# ==================================================
w2v_model = Word2Vec.load(W2V_PATH)
vocab = w2v_model.wv.key_to_index
logger.info(
    f"Word2Vec loaded from {W2V_PATH} "
    f"(vocab size={len(vocab)}, vector size={w2v_model.vector_size})"
)

def tokens_to_indices(tokens):
    idxs = [vocab[t] for t in tokens if t in vocab]
    return (idxs + [0] * SEQ_LEN)[:SEQ_LEN]

X = torch.tensor([tokens_to_indices(t) for t in tokenized], dtype=torch.long)
loader = DataLoader(X, batch_size=BATCH_SIZE)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================
model = MultiLayerLSTM(
    w2v_model,
    LSTM_HIDDEN_DIM,
    NUM_LSTM_LAYERS,
    NUM_CLASSES,
    LSTM_DROPOUT,
    FC_DROPOUT,
    BIDIRECTIONAL
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
logger.info("Model loaded successfully")

# ==================================================
# RUN INFERENCE
# ==================================================
all_preds = []

with torch.no_grad():
    for X_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

# ==================================================
# SAVE RESULTS
# ==================================================
df["predicted_label"] = [p + 1 for p in all_preds]
logger.info(f"Example inferences: {df.head(10)}")
output_path = os.path.join(DATA_DIR, "inference_results.csv")
df.to_csv(output_path, index=False)
logger.info(f"Inference results saved to {output_path}")

logger.info("==== INFERENCE END ====")

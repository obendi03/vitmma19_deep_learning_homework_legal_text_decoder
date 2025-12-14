import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
from config import *
import config
from model import  MultiLayerLSTM
from utils import setup_logger, log_config, log_model_summary, compute_class_weights, augment_legal_hungarian
# ==================================================
# LOGGER
# ==================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
logger = setup_logger("full_dataset_trainval.log")
logger.info("\n\n==== FULL DATASET TRAIN/VAL START ====")

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ==================================================
# LOAD TRAIN/VAL CSVs
# ==================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).dropna(subset=["text","rating"])
if AUGMENTATION:
    print("Alkalmazott jogi augmentáció...")
    train_df['text'] = train_df['text'].apply(lambda x: augment_legal_hungarian(x, aug_prob=AUG_PROB))
    print("Augmentáció kész!")

val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).dropna(subset=["text","rating"])
logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# ==================================================
# TOKENIZE
# ==================================================
def tokenize(texts):
    return [str(t).split()[:SEQ_LEN] for t in texts]

train_tokens = tokenize(train_df["text"].tolist())
val_tokens   = tokenize(val_df["text"].tolist())

# ==================================================
# Word2Vec EMBEDDINGS (TIF-DIF baseline)
# ==================================================
w2v_model = Word2Vec(sentences=train_tokens, vector_size=WORD2_VEC_VECTOR_SIZE,
                     min_count=WORD2_VEC_MIN_COUNT, workers=4)
vocab = w2v_model.wv.key_to_index
logger.info(f"Word2Vec trained. Vocab size: {len(vocab)} Embedding dim: {w2v_model.vector_size}")

# ==================================================
# TOKENS TO INDICES
# ==================================================
def tokens_to_indices(tokens):
    idxs = [vocab[t] for t in tokens if t in vocab]
    return (idxs + [0]*SEQ_LEN)[:SEQ_LEN]

X_train = torch.tensor([tokens_to_indices(t) for t in train_tokens], dtype=torch.long)
y_train = torch.tensor(train_df["rating"].values - 1, dtype=torch.long)
X_val   = torch.tensor([tokens_to_indices(t) for t in val_tokens], dtype=torch.long)
y_val   = torch.tensor(val_df["rating"].values - 1, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# Log config
log_config(logger, config)

# ==================================================
# MODEL
# ==================================================

model = MultiLayerLSTM(w2v_model, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS, NUM_CLASSES,
                       LSTM_DROPOUT, FC_DROPOUT,BIDIRECTIONAL).to(device)

log_model_summary(model, logger)

criterion =  nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

# ==================================================
# TRAIN LOOP
# ==================================================
best_val_loss = float("inf")
patience_counter = 0
for epoch in range(1, NUM_EPOCHS+1):
    # -------------------------------
    # TRAIN
    # -------------------------------
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # -------------------------------
    # VALIDATION
    # -------------------------------
    model.eval()
    val_loss_total = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_total += loss.item() * xb.size(0)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += xb.size(0)

    val_loss = val_loss_total / val_total
    val_acc = val_correct / val_total

    logger.info(f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # -------------------------------
    # Best model saving
    # -------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"Validation loss improved. Model saved to {MODEL_PATH}")
    else:
        patience_counter += 1
        logger.info(f"No improvement in val loss for {patience_counter} epoch(s).")

    if patience_counter >= EARLY_STOP_PATIENCE:
        logger.info(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
        break

# Save Word2Vec model
w2v_model.save(W2V_PATH)
logger.info(f"Word2Vec model saved to {os.path.join(DATA_DIR, 'w2v.model')}")
logger.info("==== FULL DATASET TRAIN/VAL END ====")

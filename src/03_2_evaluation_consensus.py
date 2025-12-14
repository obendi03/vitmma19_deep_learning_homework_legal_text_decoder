# 04_inference.py
import os
import torch
import torch.nn as nn
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from utils import setup_logger
from config import *
from model import MultiLayerLSTM
# ==================================================
# LOGGER
# ==================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
logger = setup_logger(LOG_FILE)
logger.info("\n\n==== EVALUATION CONSENSUS ====")

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ==================================================
# LOAD TEST DATA
# ==================================================
test_csv_path = os.path.join(DATA_DIR, "inference.csv")
if not os.path.exists(test_csv_path):
    logger.error(f"Inference CSV not found at {test_csv_path}")
    raise FileNotFoundError(test_csv_path)

test_df = pd.read_csv(test_csv_path).dropna(subset=["text", "rating"])
logger.info(f"Loaded {len(test_df)} samples from {test_csv_path}")

tokenized_test = [str(t).split()[:SEQ_LEN] for t in test_df["text"].tolist()]

# ==================================================
# LOAD WORD2VEC
# ==================================================

if not os.path.exists(W2V_PATH):
    logger.error(f"Word2Vec model not found at {W2V_PATH}")
    raise FileNotFoundError(W2V_PATH)

w2v_model = Word2Vec.load(W2V_PATH)
vocab = w2v_model.wv.key_to_index
logger.info(
    f"Word2Vec loaded from {W2V_PATH} "
    f"(vocab size={len(vocab)}, vector size={w2v_model.vector_size})"
)

logger.info(f"Word2Vec loaded: vocab size={len(vocab)}, vector size={w2v_model.vector_size}")

def tokens_to_indices(tokens):
    idxs = [vocab[t] for t in tokens if t in vocab]
    while len(idxs) < SEQ_LEN:
        idxs.append(0)
    return idxs[:SEQ_LEN]

X_test = torch.tensor([tokens_to_indices(t) for t in tokenized_test], dtype=torch.long)
y_test = torch.tensor(test_df["rating"].values - 1, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==================================================
# LOAD TRAINED LSTM
# ==================================================

if not os.path.exists(MODEL_PATH):
    logger.error(f"{MODEL_NAME} model not found at {MODEL_PATH}")
    raise FileNotFoundError(MODEL_PATH)

model  = MultiLayerLSTM(w2v_model, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS, NUM_CLASSES,
                       LSTM_DROPOUT, FC_DROPOUT,BIDIRECTIONAL).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
logger.info(f"{MODEL_NAME} model loaded successfully from {MODEL_PATH}")

# ==================================================
# EVALUATE LSTM
# ==================================================
criterion = nn.CrossEntropyLoss()
all_preds, all_labels = [], []

logger.info("Starting LSTM evaluation on CONSENSUS...")
test_loss_total = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(test_loader, 1):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        batch_correct = (preds == y_batch).sum().item()
        test_correct += batch_correct
        test_loss_total += loss.item() * X_batch.size(0)
        test_total += X_batch.size(0)

        logger.debug(f"Batch {i}: size={X_batch.size(0)}, loss={loss.item():.4f}, acc={batch_correct/X_batch.size(0)*100:.2f}%")

test_loss = test_loss_total / test_total
test_acc = test_correct / test_total
test_f1 = f1_score(all_labels, all_preds, average="weighted")
conf_mat = confusion_matrix(all_labels, all_preds)

logger.info(f"{MODEL_NAME} CONSENSUS Loss: {test_loss:.4f}")
logger.info(f"{MODEL_NAME} CONSENSUS Accuracy: {test_acc*100:.2f}%")
logger.info(f"{MODEL_NAME} F1-score (weighted): {test_f1:.4f}")
logger.info(f"{MODEL_NAME} Confusion Matrix:\n{conf_mat}")
logger.info(f"{MODEL_NAME} Classification Report:\n{classification_report(all_labels, all_preds)}")

# ==================================================
# TF-IDF + LOGISTIC REGRESSION BASELINE
# ==================================================
logger.info("Evaluating TF-IDF + Logistic Regression baseline...")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
if not os.path.exists(train_csv_path):
    logger.warning(f"Train CSV not found at {train_csv_path}. Skipping TF-IDF baseline.")
else:
    train_df = pd.read_csv(train_csv_path).dropna(subset=["text", "rating"])
    X_train_texts = train_df["text"].tolist()
    y_train_labels = train_df["rating"].astype(int) - 1
    logger.info(f"Loaded {len(X_train_texts)} training samples for TF-IDF baseline.")

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train_labels)
    logger.info("TF-IDF + Logistic Regression trained.")

    X_test_texts = test_df["text"].tolist()
    y_test_labels = test_df["rating"].astype(int) - 1
    X_test_tfidf = tfidf_vectorizer.transform(X_test_texts)

    y_pred = clf.predict(X_test_tfidf)
    acc_baseline = accuracy_score(y_test_labels, y_pred)
    f1_baseline = f1_score(y_test_labels, y_pred, average="weighted")
    conf_mat_baseline = confusion_matrix(y_test_labels, y_pred)

    logger.info(f"CONSENSUS: TF-IDF + Logistic Regression Accuracy: {acc_baseline*100:.2f}%")
    logger.info(f"CONSENSUS: TF-IDF + Logistic Regression F1-score (weighted): {f1_baseline:.4f}")
    logger.info(f"CONSENSUS_ TF-IDF Confusion Matrix:\n{conf_mat_baseline}")
    logger.info("CONSENSUS: TF-IDF Classification Report:\n" + classification_report(y_test_labels, y_pred))

logger.info("==== EVALUATION CONSENSUS END ====")

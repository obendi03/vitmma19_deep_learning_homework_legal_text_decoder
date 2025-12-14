import os
import torch
import torch.nn as nn
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import TensorDataset, DataLoader
from config import *
from utils import setup_logger
from model import MultiLayerLSTM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score,confusion_matrix
# ==================================================
# LOGGER
# ==================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
logger = setup_logger(LOG_FILE)
logger.info("\n\n==== EVALUATION START ====")

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA capability: "
                f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

# ==================================================
# LOAD TEST DATA
# ==================================================
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text", "rating"])
logger.info(f"Loaded test dataset with {len(test_df)} samples.")

tokenized_test = [str(t).split()[:SEQ_LEN] for t in test_df["text"].tolist()]

# ==================================================
# LOAD WORD2VEC
# ==================================================
w2v_model = Word2Vec.load(W2V_PATH)
logger.info(
    f"Word2Vec model loaded from {W2V_PATH} "
    f"(vocab size={len(w2v_model.wv)}, vector size={w2v_model.vector_size})"
)
vocab = w2v_model.wv.key_to_index


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
# LOAD LSTM MODEL
# ==================================================
model = MultiLayerLSTM(w2v_model, LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS, NUM_CLASSES,
                       LSTM_DROPOUT, FC_DROPOUT,BIDIRECTIONAL).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
logger.info(f"LSTM model loaded from {MODEL_PATH} and ready for evaluation.")

# ==================================================
# EVALUATE LSTM
# ==================================================
criterion = nn.CrossEntropyLoss()
all_preds = []
all_labels = []

test_loss_total = 0
test_correct = 0
test_total = 0

logger.info("Starting LSTM evaluation on test set...")
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        batch_correct = (preds == y_batch).sum().item()
        test_loss_total += loss.item() * X_batch.size(0)
        test_correct += batch_correct
        test_total += X_batch.size(0)
        logger.debug(f"Batch size: {X_batch.size(0)} | Batch Loss: {loss.item():.4f} | "
                     f"Batch Acc: {batch_correct / X_batch.size(0) * 100:.2f}%")

test_loss = test_loss_total / test_total
test_acc = test_correct / test_total
test_f1 = f1_score(all_labels, all_preds, average="weighted")
conf_mat = confusion_matrix(all_labels, all_preds)

logger.info(f"LSTM Test Loss: {test_loss:.4f}")
logger.info(f"LSTM Test Accuracy: {test_acc*100:.2f}%")
logger.info(f"LSTM F1-score (weighted): {test_f1:.4f}")
logger.info(f"LSTM Confusion Matrix:\n{conf_mat}")
logger.info("LSTM Classification Report:\n" + classification_report(all_labels, all_preds))

# ==================================================
# TF-IDF + Logistic Regression BASELINE
# ==================================================
logger.info("Evaluating TF-IDF + Logistic Regression baseline...")

# --- TF-IDF + Logistic Regression baseline ---
logger.info("Evaluating TF-IDF + Logistic Regression baseline...")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path).dropna(subset=["text", "rating"])
    X_train_texts = train_df["text"].tolist()
    y_train_labels = train_df["rating"].astype(int) - 1
    logger.info(f"Train dataset loaded: {len(X_train_texts)} samples.")

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

    logger.info(f"TF-IDF + Logistic Regression Test Accuracy: {acc_baseline*100:.2f}%")
    logger.info(f"TF-IDF + Logistic Regression F1-score (weighted): {f1_baseline:.4f}")
    logger.info(f"TF-IDF + Logistic Regression Confusion Matrix:\n{conf_mat_baseline}")  # ✅ Add this line
    logger.info("TF-IDF Classification Report:\n" + classification_report(y_test_labels, y_pred))
else:
    logger.warning(f"Train CSV not found at {train_csv_path}. Skipping TF-IDF baseline.")

logger.info("==== EVALUATION END ====")

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
from ray import tune
from ray.train import session
from ray.tune.schedulers import ASHAScheduler
import config
from config import *
from model import MultiLayerLSTM
from utils import setup_logger, log_config, log_model_summary, augment_legal_hungarian

# ==================================================
# LOGGER (Ray will capture stdout)
# ==================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
logger = setup_logger()
logger.info("==== RAY TUNE TRAINING START ====")

# ==================================================
# DEVICE
# ==================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ==================================================
# LOAD TRAIN / VAL DATA (ONCE)
# ==================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).dropna(subset=["text", "rating"])
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).dropna(subset=["text", "rating"])

if AUGMENTATION:
    logger.info("Applying legal Hungarian augmentation...")
    train_df["text"] = train_df["text"].apply(lambda x: augment_legal_hungarian(x, aug_prob=AUG_PROB))

logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# ==================================================
# TOKENIZATION
# ==================================================
def tokenize(texts):
    return [str(t).split()[:SEQ_LEN] for t in texts]

train_tokens = tokenize(train_df["text"].tolist())
val_tokens   = tokenize(val_df["text"].tolist())

# ==================================================
# WORD2VEC (TRAINED ON TRAIN SET ONLY)
# ==================================================
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=WORD2_VEC_VECTOR_SIZE,
    min_count=WORD2_VEC_MIN_COUNT,
    workers=4
)

vocab = w2v_model.wv.key_to_index
logger.info(f"Word2Vec vocab size: {len(vocab)} | dim: {w2v_model.vector_size}")

# ==================================================
# TOKENS → INDICES
# ==================================================
def tokens_to_indices(tokens):
    idxs = [vocab[t] for t in tokens if t in vocab]
    return (idxs + [0] * SEQ_LEN)[:SEQ_LEN]

X_train = torch.tensor([tokens_to_indices(t) for t in train_tokens], dtype=torch.long)
y_train = torch.tensor(train_df["rating"].values - 1, dtype=torch.long)
X_val   = torch.tensor([tokens_to_indices(t) for t in val_tokens], dtype=torch.long)
y_val   = torch.tensor(val_df["rating"].values - 1, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# ==================================================
# TRAIN FUNCTION FOR RAY TUNE
# ==================================================
def train_lstm_tune(cfg):
    model = MultiLayerLSTM(
        w2v_model=w2v_model,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=NUM_CLASSES,
        lstm_dropout=cfg["lstm_dropout"],
        fc_dropout=cfg["fc_dropout"],
        bidirectional=cfg["bidirectional"],
    ).to(DEVICE)

    log_model_summary(model,logger)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss /= total
        train_acc = correct / total

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # 🔥 REPORT TO RAY
        tune.report({
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc
        })


        # ---------------- EARLY STOP ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

# ==================================================
# RAY TUNE CONFIG
# ==================================================
search_space = {
    "hidden_dim": tune.choice([32, 64, 128]),
    "lr": tune.loguniform(1e-4, 3e-3),
    "lstm_dropout": tune.uniform(0.0, 0.4),
    "fc_dropout": tune.uniform(0.2, 0.6),
    "bidirectional": tune.choice([True, False]),
    "num_layers": tune.choice([1, 2, 3]),
    "epochs": NUM_EPOCHS,
}

scheduler = ASHAScheduler(
    metric="val_acc",
    mode="max",
    max_t=NUM_EPOCHS,
    grace_period=2,
    reduction_factor=2,
)
# ==================================================
# RUN RAY TUNE
# ==================================================
if __name__ == "__main__":
    #log_config(logger, config)

    tune.run(
        train_lstm_tune,
        config=search_space,
        num_samples=20,
        scheduler=None,  #
        resources_per_trial={"gpu": 1},
        stop={"training_iteration": 1000},  # ✅ MINDEN trial 1000 iteráció!
    )

    logger.info("==== RAY TUNE TRAINING END ====")
    logger.info("==== RAY TUNE TRAINING END ====")
# -------------------------------------
# Paths
# -------------------------------------
DATA_DIR = "/app/data"
LOG_FILE = "/app/logs/run.log"
OUTPUT_DIR = "/app/output"
MODEL_PATH = f"{OUTPUT_DIR}/lstm_model.pth"
W2V_PATH   = f"{OUTPUT_DIR}/w2v.model"
MODEL_NAME = "LSTM"
# -------------------------------------
# Training Hyperparameters
# -------------------------------------

BATCH_SIZE = 32
SEQ_LEN = 500
NUM_CLASSES = 5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1000

LSTM_HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
FC_DROPOUT = 0.4
BIDIRECTIONAL = False
LEARNING_RATE = 0.001

#best hpo params top1
# LSTM_HIDDEN_DIM = 128
# NUM_LSTM_LAYERS = 1
# LSTM_DROPOUT = 0.139
# FC_DROPOUT = 0.515
# BIDIRECTIONAL = True
# LEARNING_RATE = 0.00021


# ========================================
# 🥈 #2 (39.77% val_acc) - trial_00011
# ========================================
# LSTM_HIDDEN_DIM = 32
# NUM_LSTM_LAYERS = 3
# LSTM_DROPOUT = 0.096
# FC_DROPOUT = 0.216
# BIDIRECTIONAL = True
# LEARNING_RATE = 0.000163

# ========================================
# 🥉 #3 (40.07% val_acc) - trial_00017
# ========================================
# LSTM_HIDDEN_DIM = 64
# NUM_LSTM_LAYERS = 2
# LSTM_DROPOUT = 0.299
# FC_DROPOUT = 0.492
# BIDIRECTIONAL = False
# LEARNING_RATE = 0.000136

AUGMENTATION = True
AUG_PROB =  0.15

# WORD2VEC PARAMETERS
WORD2_VEC_VECTOR_SIZE = 64
WORD2_VEC_MIN_COUNT = 1

#

# -------------------------------------
# Early stopping
# -------------------------------------
EARLY_STOP_PATIENCE = 20  # number of epochs with no improvement before stopping
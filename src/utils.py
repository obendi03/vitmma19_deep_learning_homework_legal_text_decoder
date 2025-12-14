import logging
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import random

def setup_logger(log_file=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    # stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_model_summary(model, logger):
    logger.info("==== MODEL SUMMARY START ====")

    # Full architecture
    logger.info("Model architecture:")
    logger.info(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Total parameters      : {total_params:,}")
    logger.info(f"Trainable parameters  : {trainable_params:,}")
    logger.info(f"Frozen parameters     : {frozen_params:,}")

    # Optional: embedding breakdown
    if hasattr(model, "emb"):
        emb_params = model.emb.weight.numel()
        logger.info(f"Embedding parameters  : {emb_params:,}")

    logger.info("==== MODEL SUMMARY END ====")

def log_config(logger, config_module):
    logger.info("==== CONFIG & HYPERPARAMETERS START ====")

    for key in dir(config_module):
        if key.isupper():
            value = getattr(config_module, key)
            logger.info(f"{key:20s}: {value}")

    logger.info("==== CONFIG & HYPERPARAMETERS END ====")


def compute_class_weights(train_labels, method='balanced', smoothing=0.1):
    """
    Automatic class weight calculation for PyTorch.

    Args:
        train_labels: numpy array or list of labels (0-4)
        method: 'balanced' (sklearn) or 'inverse_freq'
        smoothing: 0.0-1.0 (prevents infinite weights for zero classes)

    Returns:
        torch.Tensor: class weights for CrossEntropyLoss
    """
    # Count frequencies
    class_counts = Counter(train_labels)
    n_classes = len(set(train_labels))
    n_samples = len(train_labels)

    print(f"Class distribution: {dict(sorted(class_counts.items()))}")

    if method == 'balanced':
        # sklearn method: n_samples / (n_classes * class_count)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
    else:  # inverse frequency
        weights = [n_samples / (n_classes * class_counts.get(i, 1)) for i in range(n_classes)]

    # Smoothing (prevents NaN for missing classes)
    weights = np.array(weights) + smoothing

    # Normalize
    weights = weights / weights.sum() * n_classes

    print(f"Raw weights: {weights}")
    print(f"PyTorch weights: {torch.tensor(weights)}")

    return torch.tensor(weights, dtype=torch.float).to('cuda')  # Adjust device as needed

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def augment_legal_hungarian(text, aug_prob=0.15):
    """Magyar jogi szöveg augmentáció szinonimákkal"""
    legal_synonyms = {
        # Szerződések
        "szerződés": ["megállapodás", "egyezség", "alku"],
        "szerződő": ["felek", "oldal", "részes"],
        "szerződő fél": ["felek", "szerződő partner"],

        # Kötelezettségek
        "kötelezettség": ["kötelezés", "elkötelezettség", "kötelesség"],
        "megsért": ["megtörténik", "sért", "megtörténik"],
        "megsértése": ["sértése", "megtörténik"],

        # Jogok
        "jog": ["jogosultság", "hatalom", "jogosítvány"],
        "jogosultság": ["jog", "felhatalmazás"],
        "felhatalmazás": ["jogosultság", "engedély"],

        # Bíróság
        "bíróság": ["törvényszék", "perelő"],
        "ítélet": ["döntés", "határozat"],
        "per": ["perek", "hivatalos eljárás"],

        # Egyéb gyakori jogi kifejezések
        "kártérítés": ["kárpótlás", "megtérítés"],
        "felelősség": ["elkötelezettség", "felelősségre vonás"],
        "megállapodás": ["szerződés", "egyezség"],
        "hatályos": ["érvényes", "hatályban lévő"],
        "érvénytelen": ["nulla", "hatálytalan"]
    }

    words = text.lower().split()
    augmented_words = []

    for word in words:
        if random.random() < aug_prob and word in legal_synonyms:
            augmented_words.append(random.choice(legal_synonyms[word]))
        else:
            augmented_words.append(word)

    return " ".join(augmented_words)
import torch
from torch import nn as nn


class MultiLayerLSTM(nn.Module):
    def __init__(self, w2v_model, hidden_dim, num_layers, num_classes, lstm_dropout=0, fc_dropout=0,
                 bidirectional=False):
        super().__init__()
        vocab_size, embed_dim = w2v_model.wv.vectors.shape
        self.emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.emb.weight.data[1:vocab_size + 1] = torch.tensor(w2v_model.wv.vectors, dtype=torch.float)

        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers,
                            batch_first=True,
                            dropout=lstm_dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)

        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        # Embed
        embedded = self.emb(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)

        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Final layers
        output = self.fc(self.dropout(context))
        return output
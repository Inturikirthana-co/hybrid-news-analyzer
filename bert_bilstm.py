# bert_bilstm.py
"""
BERT + BiLSTM classifier for binary text classification (Real vs Fake).
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BertBiLSTM(nn.Module):
    def __init__(self,
                 encoder_name="bert-base-uncased",
                 lstm_hidden_dim=256,
                 lstm_layers=1,
                 bidirectional=True,
                 dropout=0.3,
                 num_labels=2):
        """
        encoder_name: pretrained transformer to use (HuggingFace model name).
        lstm_hidden_dim: hidden size for the BiLSTM.
        lstm_layers: number of LSTM layers.
        bidirectional: use BiLSTM (True).
        dropout: dropout before final classification.
        num_labels: number of output classes (2 -> real/fake).
        """
        super(BertBiLSTM, self).__init__()

        # Pretrained transformer encoder (returns last hidden states)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # Size of transformer hidden states (e.g., 768 for bert-base)
        encoder_hidden_size = self.encoder.config.hidden_size

        # BiLSTM that reads encoder token embeddings
        self.lstm = nn.LSTM(input_size=encoder_hidden_size,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0.0)

        # If bidirectional, LSTM output dim = hidden * 2
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids, attention_mask: tensors from tokenizer.
        Returns raw logits (before softmax).
        """
        # Get transformer outputs (last_hidden_state shape: batch, seq_len, hidden)
        encoder_outputs = self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state  # (B, L, H)

        # Pass token embeddings to LSTM
        # lstm_out: (B, L, hidden*directions)
        lstm_out, _ = self.lstm(last_hidden_state)

        # We can pool LSTM outputs. Options: last timestep, max-pool, mean-pool.
        # Using mean pooling over sequence length (masked) is generally robust.
        # However attention_mask helps compute masked mean:
        # compute masked mean of lstm_out across seq_len
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        lstm_out_masked = lstm_out * mask  # zero out padded tokens

        # sum over tokens and divide by number of valid tokens
        denom = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        pooled = lstm_out_masked.sum(dim=1) / denom  # (B, hidden*directions)

        # Pass pooled vector to classifier head
        logits = self.classifier(pooled)  # (B, num_labels)
        return logits

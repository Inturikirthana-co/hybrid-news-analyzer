# train_fake_news.py
"""
Training script for BertBiLSTM fake-news classifier.

Expected dataset format:
A CSV file with 'text' and 'label' columns (label: 0 for REAL, 1 for FAKE).
"""

import os
import argparse
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from bert_bilstm import BertBiLSTM

# ------- simple Dataset wrapper -------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def eval_model(model, dataloader, device):
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            truths.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(truths, preds)
    p, r, f1, _ = precision_recall_fscore_support(truths, preds, average="binary")
    return acc, p, r, f1


def read_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    # dropna and keep only text and label columns
    df = df.dropna(subset=["text", "label"])
    return df["text"].tolist(), df["label"].tolist()


def main(args):
    # Repro
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = BertBiLSTM(encoder_name=args.encoder_name,
                       lstm_hidden_dim=args.lstm_hidden_dim,
                       lstm_layers=args.lstm_layers,
                       bidirectional=args.bidirectional,
                       dropout=args.dropout,
                       num_labels=2)
    model.to(device)

    # Load dataset
    texts, labels = read_csv(args.data_csv)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer & scheduler (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps),
                                                num_training_steps=total_steps)

    # Training loop
    best_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        val_acc, val_p, val_r, val_f1 = eval_model(model, val_loader, device)
        print(f"Validation — Acc: {val_acc:.4f}, P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(args.output_dir, "bert_bilstm_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_name": args.encoder_name,
                "args": vars(args)
            }, save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True,
                        help="path to CSV with 'text' and 'label' columns")
    parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lstm_hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="saved_models")
    args = parser.parse_args()
    main(args)

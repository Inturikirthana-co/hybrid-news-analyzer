import torch
from transformers import BertTokenizer, pipeline
from flask import Flask, render_template, request
from bert_bilstm import BertBiLSTM
from news_ai_model import check_google_semantic_similarity
import gc
from functools import lru_cache

# -------------------------------------------------
# 1️⃣ Initialize Flask App
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# 2️⃣ Load Tokenizer and Model
# -------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL_PATH = "saved_models/bert_bilstm_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fake_news_model = BertBiLSTM()
checkpoint = torch.load(MODEL_PATH, map_location=device)
fake_news_model.load_state_dict(checkpoint["model_state_dict"])
fake_news_model.to(device)
fake_news_model.eval()

print("✅ Model loaded successfully from", MODEL_PATH)

# -------------------------------------------------
# 3️⃣ Cached lightweight models for summarization & sentiment
# -------------------------------------------------
@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# -------------------------------------------------
# 4️⃣ Fake News Prediction Function
# -------------------------------------------------
def predict_news(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = fake_news_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "REAL" if pred == 0 else "FAKE"
    confidence = probs[0][pred].item()

    # 🧹 Free up memory for Render
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label, confidence

# -------------------------------------------------
# 5️⃣ Flask Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news_text = request.form["news_text"]

        # --- Fake News Detection ---
        label, confidence = predict_news(news_text)

        # --- Google / SERPAPI Verification ---
        verification_status, similar_articles = check_google_semantic_similarity(news_text)

        if verification_status == "Verified":
            label = "REAL ✅ Verified via Google"
        elif verification_status == "Unverified":
            label = "FAKE ❌ Not Verified by Google"

        # --- Summarization ---
        try:
            summarizer = get_summarizer()
            summary = summarizer(news_text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
        except Exception:
            summary = "Text too short or invalid for summarization."

        # --- Sentiment Analysis ---
        try:
            sentiment_analyzer = get_sentiment_analyzer()
            sentiment_result = sentiment_analyzer(news_text)[0]
            sentiment = sentiment_result['label']
            sentiment_score = sentiment_result['score']
        except Exception:
            sentiment = "N/A"
            sentiment_score = 0

        return render_template(
            "index.html",
            news_text=news_text,
            label=label,
            confidence=round(confidence * 100, 2),
            summary=summary,
            sentiment=sentiment,
            sentiment_score=round(sentiment_score * 100, 2),
            similar_articles=similar_articles
        )

    return render_template("index.html")

# -------------------------------------------------
# 6️⃣ Run Flask App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

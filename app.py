import torch
from transformers import BertTokenizer, pipeline
from flask import Flask, render_template, request
import gc
from news_ai_model import check_google_semantic_similarity
from functools import lru_cache

app = Flask(__name__)

# -------------------------------------------------
# 1️⃣ Lazy-load model only when needed
# -------------------------------------------------
@lru_cache(maxsize=1)
def load_fake_news_model():
    from bert_bilstm import BertBiLSTM
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertBiLSTM()
    checkpoint = torch.load("saved_models/bert_bilstm_best.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer

# -------------------------------------------------
# 2️⃣ Cached Summarizer & Sentiment Models
# -------------------------------------------------
@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

# -------------------------------------------------
# 3️⃣ Fake News Prediction
# -------------------------------------------------
def predict_news(text):
    model, tokenizer = load_fake_news_model()
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    label = "REAL" if pred == 0 else "FAKE"
    confidence = probs[0][pred].item()
    gc.collect()
    return label, confidence

# -------------------------------------------------
# 4️⃣ Main Route
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news_text = request.form["news_text"]

        label, confidence = predict_news(news_text)
        verification_status, similar_articles = check_google_semantic_similarity(news_text)

        if verification_status == "Verified":
            label = "REAL ✅ Verified via Google"
        elif verification_status == "Unverified":
            label = "FAKE ❌ Not Verified by Google"

        try:
            summary = get_summarizer()(news_text, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]
        except Exception:
            summary = "Text too short for summarization."

        try:
            sentiment_data = get_sentiment_analyzer()(news_text)[0]
            sentiment = sentiment_data["label"]
            sentiment_score = sentiment_data["score"]
        except Exception:
            sentiment = "N/A"
            sentiment_score = 0

        gc.collect()

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
# 5️⃣ Render Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

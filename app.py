import torch
from transformers import DistilBertTokenizer, pipeline
from flask import Flask, render_template, request
from news_ai_model import predict_fake_news_local, check_google_semantic_similarity

# -------------------------------------------------
# 1️⃣ Initialize Flask App
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# 2️⃣ Load Tokenizer and Smaller NLP Pipelines
# -------------------------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Use lightweight pre-trained models (memory-friendly)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("✅ Lightweight NLP pipelines loaded successfully!")

# -------------------------------------------------
# 3️⃣ Main Prediction Function
# -------------------------------------------------
def analyze_news(news_text):
    # --- Fake News Detection (Hybrid Model + Google Check) ---
    prediction, confidence = predict_fake_news_local(news_text)
    google_verification = check_google_semantic_similarity(news_text)

    if "similar articles found" in google_verification.lower():
        if prediction == "FAKE" and confidence < 0.9:
            prediction = "REAL ✅"
        else:
            prediction = "FAKE 🚫"
    else:
        prediction = f"{prediction} ⚠️ (No strong verification found)"

    # --- Summarization ---
    try:
        summary = summarizer(news_text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    except Exception:
        summary = "Text too short or invalid for summarization."

    # --- Sentiment Analysis ---
    try:
        sentiment_result = sentiment_analyzer(news_text)[0]
        sentiment = sentiment_result['label']
        sentiment_score = round(sentiment_result['score'] * 100, 2)
    except Exception:
        sentiment = "N/A"
        sentiment_score = 0

    return prediction, confidence, summary, sentiment, sentiment_score, google_verification


# -------------------------------------------------
# 4️⃣ Flask Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news_text = request.form["news_text"]

        prediction, confidence, summary, sentiment, sentiment_score, google_verification = analyze_news(news_text)

        return render_template(
            "index.html",
            news_text=news_text,
            label=prediction,
            confidence=round(confidence * 100, 2),
            summary=summary,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            google_verification=google_verification
        )

    return render_template("index.html")


# -------------------------------------------------
# 5️⃣ Run Flask App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

from flask import Flask, render_template, request
from transformers import pipeline
from news_ai_model import predict_fake_news_local, check_google_semantic_similarity

# ----------------------------------------------
# Initialize Flask app
# ----------------------------------------------
app = Flask(__name__)

# 🧠 Replace this with your actual SerpAPI key
SERPAPI_KEY = "a2d7f62d58e6044c76a6af0f06f180ee9b4a0bf0f04c82c44ac5e1a1a1df30d2"

# ----------------------------------------------
# Load Pretrained Pipelines
# ----------------------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# ----------------------------------------------
# Web Routes
# ----------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news_text = request.form["news_text"]

        # --- Step 1: Local Model Prediction ---
        label, confidence = predict_fake_news_local(news_text)

        # --- Step 2: Google Semantic Verification ---
        google_result, similarity = check_google_semantic_similarity(news_text, SERPAPI_KEY)

        # --- Step 3: Hybrid Fusion Logic ---
        if google_result is True and similarity > 0.6:
            final_label = "✅ REAL NEWS"
            verification = f"Verified online (Semantic Match: {similarity:.2f})"
        elif google_result is False and similarity < 0.3:
            final_label = "🚨 FAKE NEWS"
            verification = f"No reliable match found (Similarity: {similarity:.2f})"
        else:
            final_label = "⚠️ UNCERTAIN"
            verification = f"Low confidence (Semantic Match: {similarity:.2f})"

        # --- Step 4: Text Summarization ---
        try:
            summary = summarizer(
                news_text, max_length=80, min_length=20, do_sample=False
            )[0]["summary_text"]
        except Exception:
            summary = "Text too short or invalid for summarization."

        # --- Step 5: Sentiment Analysis ---
        sentiment_result = sentiment_analyzer(news_text)[0]
        sentiment = sentiment_result["label"]
        sentiment_score = round(sentiment_result["score"] * 100, 2)

        # --- Step 6: Render Result ---
        return render_template(
            "index.html",
            news_text=news_text,
            label=final_label,
            confidence=confidence,
            summary=summary,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            verification=verification
        )

    # Default home page
    return render_template("index.html")

# ----------------------------------------------
# Run Flask app
# ----------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

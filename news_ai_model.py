import torch
from transformers import AutoTokenizer
from bert_bilstm import BertBiLSTM
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer, util

# Initialize semantic search model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Fake News Model
MODEL_PATH = "saved_models/bert_bilstm_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading BERT+BiLSTM model...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
encoder_name = checkpoint.get("tokenizer_name", "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(encoder_name)

model = BertBiLSTM(
    encoder_name=encoder_name,
    lstm_hidden_dim=checkpoint.get("args", {}).get("lstm_hidden_dim", 256),
    lstm_layers=checkpoint.get("args", {}).get("lstm_layers", 1),
    bidirectional=checkpoint.get("args", {}).get("bidirectional", True),
    dropout=checkpoint.get("args", {}).get("dropout", 0.3),
    num_labels=2
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("✅ Fake-news model loaded successfully.")


def predict_fake_news_local(text, max_length=256):
    try:
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "REAL" if pred == 0 else "FAKE"
        return label, round(confidence * 100, 2)
    except Exception as e:
        print("Error during fake news prediction:", e)
        return "UNKNOWN", 0.0


def check_google_semantic_similarity(query, api_key):
    """
    Uses Google Search (SerpAPI) + SentenceTransformer for semantic similarity.
    Returns (True/False, similarity_score)
    """
    try:
        params = {
            "q": query,
            "hl": "en",
            "num": 5,
            "api_key": api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            print("❌ SerpAPI Error:", results["error"])
            return None, 0.0

        articles = results.get("organic_results", [])
        if not articles:
            print("⚠️ No articles found online.")
            return False, 0.0

        query_embed = semantic_model.encode(query, convert_to_tensor=True)
        max_similarity = 0.0

        for article in articles:
            text = (article.get("title", "") + " " + article.get("snippet", "")).strip()
            if not text:
                continue
            article_embed = semantic_model.encode(text, convert_to_tensor=True)
            similarity = float(util.pytorch_cos_sim(query_embed, article_embed))
            max_similarity = max(max_similarity, similarity)

        print(f"🔍 Max semantic similarity: {max_similarity:.2f}")

        if max_similarity > 0.6:
            return True, max_similarity
        elif max_similarity > 0.3:
            return None, max_similarity  # uncertain
        else:
            return False, max_similarity

    except Exception as e:
        print("⚠️ Google Semantic Search error:", e)
        return None, 0.0

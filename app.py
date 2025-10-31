import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from google_search_results import GoogleSearchResults

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Hybrid News Analyzer", page_icon="📰", layout="wide")

st.title("📰 Hybrid News Analyzer")
st.markdown("Analyze news sentiment and credibility using AI + SERP API")

query = st.text_input("Enter a topic or headline:")

if st.button("Analyze"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Fetching news and analyzing sentiment..."):
            try:
                # Fetch top search results via SERPAPI
                params = {"q": query, "num": 5, "api_key": "a2d7f62d58e6044c76a6af0f06f180ee9b4a0bf0f04c82c44ac5e1a1a1df30d2"}
                search = GoogleSearchResults(params)
                results = search.get_dict()

                # Display results
                if "organic_results" in results:
                    st.subheader("Top News Results:")
                    for i, item in enumerate(results["organic_results"], start=1):
                        st.markdown(f"### {i}. {item.get('title', 'No title')}")
                        st.write(item.get("snippet", "No description available."))
                        st.write(f"[Read more]({item.get('link', '')})")

                # Sentiment analysis using pretrained transformer
                st.subheader("Sentiment Analysis:")
                model = pipeline("sentiment-analysis")
                sentiment = model(query)[0]
                st.success(f"**{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")

            except Exception as e:
                st.error(f"Error: {e}")

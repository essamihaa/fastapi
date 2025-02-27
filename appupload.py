import os
import logging
import nltk
import spacy
import torch
import faiss
import numpy as np
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load spaCy for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect Device (GPU or CPU)
device = 0 if torch.cuda.is_available() else -1

# **Load Summarization Model**
try:
    logger.info("Loading summarization model...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    logger.info("✅ Summarization model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading summarization model: {e}")
    summarizer = None  # Prevent crash if model fails

# **Load Sentiment Analysis Model**
try:
    logger.info("Loading sentiment analysis model...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    logger.info("✅ Sentiment analysis model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading sentiment model: {e}")
    sentiment_analyzer = None

# **Load Embedding Model**
try:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading embedding model: {e}")
    embedder = None

# **Initialize FAISS Vector Store**
embedding_size = 384  # Based on MiniLM
index = faiss.IndexFlatL2(embedding_size)
vector_store = []  # Store text along with FAISS index

# Define Text Request Structure
class TextRequest(BaseModel):
    text: str

# Function to extract bigram themes (without scores)
def extract_themes(text, top_n=5):
    try:
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Generate bigrams
        bi_grams = list(ngrams(filtered_words, 2))
        
        # Convert bigrams to strings for counter
        bi_gram_phrases = [f"{w1} {w2}" for w1, w2 in bi_grams]
        
        # Count and return top themes (names only)
        theme_counts = Counter(bi_gram_phrases)
        top_themes = [theme for theme, _ in theme_counts.most_common(top_n)]
        
        # Ensure we return something
        if not top_themes:
            return ["other"]
        
        return top_themes
    except Exception as e:
        logger.error(f"❌ Theme extraction failed: {e}")
        return ["error", "other"]

# **📌 Main Text Analysis API**
@app.post("/analyze/")
def analyze_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    text = request.text.strip()
    logger.info(f"📥 Incoming text: {text}")

    # **Step 1: Summarization**
    summary = "Summarization model not available."
    if summarizer:
        try:
            input_length = len(text.split())
            max_length = min(100, max(10, int(input_length * 0.8)))  # Dynamic max_length
            logger.info(f"🔹 Summarizing with max_length={max_length}, min_length=10")
            summary_result = summarizer(text, max_length=max_length, min_length=10, num_beams=4, early_stopping=True)
            summary = summary_result[0]["summary_text"]
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            summary = f"Error in summarization: {str(e)}"

    # **Step 2: Named Entity Recognition (NER)**
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]

    # **Step 3: Sentiment Analysis**
    sentiment = {"error": "Sentiment model not available"}
    if sentiment_analyzer:
        try:
            sentiment_result = sentiment_analyzer(summary)[0]
            sentiment = {"label": sentiment_result["label"], "score": round(sentiment_result["score"], 2)}
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            sentiment = {"error": f"Sentiment analysis failed: {str(e)}"}

    # **Step 4: Thematic Analysis with 2-word themes (without scores)**
    themes = extract_themes(text)
    logger.info(f"🔍 Extracted themes: {themes}")

    # **Step 5: Extract Most Common Words**
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(10)  # Ensure this always exists

    # **Step 6: Store Embeddings in FAISS**
    faiss_status = "Not Stored"
    if embedder:
        try:
            embedding = embedder.encode([text])[0]  # Get vector
            index.add(np.array([embedding], dtype=np.float32))  # Add to FAISS index
            vector_store.append({"text": text, "vector": embedding})
            faiss_status = "Stored Successfully"
        except Exception as e:
            logger.error(f"❌ Failed to store data in FAISS: {e}")
            faiss_status = f"Storage Failed: {str(e)}"

    # **Final Response (Ensuring All Required Fields Exist)**
    return {
        "summary": summary,
        "themes": themes,  # Now contains just the theme names as a list
        "sentiment": sentiment,
        "most_common_words": most_common_words,
        "named_entities": named_entities,
        "faiss_status": faiss_status
    }

# **Root Endpoint**
@app.get("/")
def read_root():
    return {"message": "FastAPI with Summarization, 2-Word Thematic Analysis, Sentiment Analysis, and FAISS!"}
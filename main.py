import logging
import spacy
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from models.summarization import get_summary
from models.sentiment import analyze_sentiment
from models.embeddings import store_embedding
from models.themes import extract_themes

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Define Text Request Structure
class TextRequest(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    text = request.text.strip()
    logger.info(f"📥 Incoming text: {text}")

    summary = get_summary(text)
    sentiment = analyze_sentiment(summary)
    themes = extract_themes(text)
    named_entities = [ent.text for ent in nlp(text).ents]
    faiss_status = store_embedding(text)

    return {
        "summary": summary,
        "themes": themes,
        "sentiment": sentiment,
        "named_entities": named_entities,
        "faiss_status": faiss_status
    }

@app.get("/")
def read_root():
    return {"message": "FastAPI Text Analyzer API"}

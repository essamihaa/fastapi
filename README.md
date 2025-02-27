# 📝 FastAPI Text Analyzer 🚀

A **FastAPI-powered** text analysis API with summarization, sentiment analysis, named entity recognition (NER), and FAISS-based vector storage.

## 📌 Features
✅ **Summarization** (DistilBART CNN model)  
✅ **Sentiment Analysis** (DistilBERT model)  
✅ **Named Entity Recognition (NER)** (spaCy)  
✅ **Theme Extraction** (Bigram themes via NLTK)  
✅ **FAISS Vector Storage** (Efficient text embeddings)  

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/fastapi-text-analyzer.git
cd fastapi-text-analyzer
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the API Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ Test API Endpoints
```bash
curl -X POST "http://localhost:8000/analyze/" -H "Content-Type: application/json" -d '{"text": "Sample text for analysis."}'
```

## 🚀 Deployment
- **Docker**: `docker build -t fastapi-text-analyzer .`
- **Gunicorn (Production)**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`

## 📄 License
MIT License

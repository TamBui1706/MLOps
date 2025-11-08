
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import pickle
import re
from nltk.corpus import stopwords
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AG News Classification API",
    description="API for classifying news articles into 4 categories: World, Sports, Business, Sci/Tech",
    version="1.0.0"
)

# Load model, vectorizer, and label mapping
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_names.pkl', 'rb') as f:
    label_names = pickle.load(f)

# Define input model
class NewsArticle(BaseModel):
    text: str = Field(..., description="News article text", min_length=10)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple announces new iPhone with advanced AI features and improved camera system."
            }
        }

# Define output model
class PredictionResponse(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]

# Preprocess function
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AG News Classification API",
        "model": "Logistic Regression with TF-IDF",
        "categories": list(label_names.values()),
        "description": "This API classifies news articles into 4 categories: World, Sports, Business, and Sci/Tech",
        "endpoints": {
            "/": "Get API information",
            "/predict": "Predict category for news article (POST)",
            "/health": "Check API health status"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(article: NewsArticle):
    try:
        # Preprocess text
        text_clean = preprocess_text(article.text)
        
        if not text_clean:
            raise HTTPException(status_code=400, detail="Text is empty after preprocessing")
        
        # Vectorize
        text_tfidf = vectorizer.transform([text_clean])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf)[0]
        
        # Prepare response
        category = label_names[prediction]
        confidence = float(proba[prediction])
        probabilities = {label_names[i]: float(proba[i]) for i in range(len(proba))}
        
        return PredictionResponse(
            category=category,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

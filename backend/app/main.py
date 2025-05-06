from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import uvicorn
from pathlib import Path

app = FastAPI(title="Spam Email Detector API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Using a pre-trained model from Hugging Face instead of local model
model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

# Load model and tokenizer once during startup
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    print(f"Model loaded successfully: {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class EmailRequest(BaseModel):
    email_content: str

class PredictionResponse(BaseModel):
    is_spam: bool
    confidence: float
    details: dict

@app.get("/")
def read_root():
    return {"message": "Spam Email Detector API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_email(request: EmailRequest):
    try:
        if not request.email_content.strip():
            raise HTTPException(status_code=400, detail="Email content cannot be empty")

        # Run prediction
        result = pipe(request.email_content)[0]

        # Extract prediction and score
        label = result['label']
        score = result['score']

        # For this model, "LABEL_1" means spam (SPAM), "LABEL_0" means ham (NOT_SPAM)
        is_spam = label == "LABEL_1"

        # Create response
        response = {
            "is_spam": is_spam,
            "confidence": float(score),
            "details": {
                "raw_label": label,
                "raw_score": float(score),
                "model_name": model_name,
                "content_length": len(request.email_content)
            }
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification
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

# Get the absolute path to the saved-model directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.path.join(BASE_DIR, "saved-model")

# Load model and tokenizer once during startup
try:
    print(f"Loading model from: {MODEL_DIR}")
    try:
        # Try to load from local directory first
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("Model and tokenizer loaded successfully from local directory")
    except Exception as local_error:
        print(f"Error loading from local directory: {local_error}")
        print("Falling back to loading tokenizer from Hugging Face and model from local directory")
        # Load tokenizer from HuggingFace and model from local saved files
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("Successfully loaded tokenizer from Hugging Face and model from local directory")

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    print("TextClassificationPipeline created successfully")
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
                "model_path": str(MODEL_DIR),
                "content_length": len(request.email_content)
            }
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

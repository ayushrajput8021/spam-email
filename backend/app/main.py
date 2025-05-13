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
MODEL_HF_ID = "distilbert-base-uncased"  # Fallback model ID from HuggingFace

# Load model and tokenizer once during startup
try:
    print(f"Loading model from: {MODEL_DIR}")

    # Strategy 1: Try to load both tokenizer and model from local directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("SUCCESS: Model and tokenizer loaded successfully from local directory")
    except Exception as local_error:
        print(f"PARTIAL FAILURE: Error loading from local directory: {local_error}")

        # Strategy 2: Try loading tokenizer from HuggingFace and model from local directory
        try:
            print(f"Trying fallback strategy: tokenizer from HuggingFace ({MODEL_HF_ID}) and model from local")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            print("SUCCESS: Loaded tokenizer from HuggingFace and model from local directory")
        except Exception as mixed_error:
            print(f"PARTIAL FAILURE: Mixed loading strategy failed: {mixed_error}")

            # Strategy 3: Full fallback to HuggingFace for both
            try:
                print(f"Trying complete fallback: loading both tokenizer and model from HuggingFace ({MODEL_HF_ID})")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF_ID)
                print("SUCCESS: Loaded both tokenizer and model from HuggingFace")
            except Exception as hf_error:
                print(f"COMPLETE FAILURE: All loading strategies failed. Last error: {hf_error}")
                raise hf_error

    # Create the classification pipeline
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

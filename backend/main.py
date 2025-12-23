from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import torch
import os
import shutil
from datetime import datetime

from .model import MentalHealthModel
from .preprocess import preprocess_text, preprocess_audio
from .database import SessionLocal, init_db, PredictionRecord

app = FastAPI(title="Mental Health Risk Score Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize DB
init_db()

# Load Model
model = MentalHealthModel()
# try:
#     model.load_state_dict(torch.load("model.pth"))
#     model.eval()
# except:
#     print("Model not found, using initialized weights")

@app.get("/")
def read_root():
    return {"message": "Mental Health Risk Assessment API is running"}

def save_prediction(db: Session, text: str, audio_path: str, score: float):
    risk_level = "High" if score > 0.7 else "Moderate" if score > 0.3 else "Low"
    db_record = PredictionRecord(
        text_content=text,
        audio_path=audio_path,
        risk_score=score,
        risk_level=risk_level
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

@app.post("/analyze/multimodal")
async def analyze_multimodal(
    text: str = Form(...),
    age: int = Form(...),
    sleep_quality: float = Form(...),
    stress_level: float = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save audio file
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    file_location = f"{upload_dir}/{datetime.now().timestamp()}_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Preprocess
    text_features = preprocess_text(text)
    # in real app, load audio from file_location
    audio_features = preprocess_audio(None) 
    
    # Simple heuristic to combine tabular data with model (Hybrid)
    # In a real system, we would feed these into the Neural Net
    tabular_risk_factor = (stress_level * 0.4) + ((10 - sleep_quality) * 0.1)
    
    # Inference
    with torch.no_grad():
        base_risk = model(text_features, audio_features).item()
    
    # Weighted Average
    final_score = (base_risk * 0.7) + (tabular_risk_factor * 0.3)
    final_score = min(max(final_score, 0.0), 1.0) # Clip
    
    # Store Data
    save_prediction(db, text, file_location, final_score)
    
    return {
        "risk_score": final_score, 
        "modality": "multimodal",
        "saved": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

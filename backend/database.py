from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./mental_health.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Input Data
    text_content = Column(Text, nullable=True)
    audio_path = Column(String, nullable=True)
    
    # Analysis Results
    risk_score = Column(Float)
    risk_level = Column(String)
    
    # User Feedback (Optional for future)
    user_feedback = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

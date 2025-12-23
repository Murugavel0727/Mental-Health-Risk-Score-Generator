# Mental Health Risk Score Generator

## Overview
A Deep Learning project that analyzes Journal Entries (Text) and Voice Recordings (Audio) to assess mental health risk.

## Project Structure
- `backend/`: FastAPI server + PyTorch Model
- `frontend/`: Premium HTML/CSS/JS Interface
- `ml/`: Model training scripts and dataset definitions

## How to Run

### 1. Backend
1. Open a terminal in `backend/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   *Note: Ensure you are in the `backend` directory or adjust path in command.*

### 2. Frontend
1. Simply open `frontend/index.html` in your web browser.
2. The frontend connects to `http://localhost:8000` by default.

## Features
- **Multimodal Analysis**: Combines Text and Audio.
- **Privacy-First**: Analysis happens on the backend model (currently mocked for demo).
- **Glassmorphism UI**: Modern, calming interface.

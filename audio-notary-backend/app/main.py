from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth, analyze
import os

app = FastAPI()

# --- CORS CONFIGURATION (Crucial for Vercel) ---
origins = [
    "http://localhost:5173",                          # Localhost Development
    "http://127.0.0.1:5173",                          # Localhost Alternative
    "https://audio-deepfake-detector-jet.vercel.app", # <--- YOUR VERCEL FRONTEND
    "https://audio-deepfake-detector-jet.vercel.app/" # Trailing slash version
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Explicitly allow your Vercel app
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers (Authorization, etc.)
)

# Register Routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])

@app.get("/")
def read_root():
    return {"message": "Audio Notary Backend is Running Securely"}
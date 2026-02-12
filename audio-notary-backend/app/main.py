from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import auth, analyze

app = FastAPI()

# Enable CORS for Frontend
origins = [
    "http://localhost:5173",           # Localhost
    "https://audio-deepfake-detector-jet.vercel.app/" # <--- You will add your Vercel URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Keep "*" for now to make deployment easier initially
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])

@app.get("/")
def read_root():
    return {"message": "Audio Notary Backend is Running"}
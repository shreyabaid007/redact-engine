import os
import logging
import tempfile
import hmac
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from modal import Image, Secret, App, gpu, method, Volume, asgi_app

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# App and Volume Setup
app = App("modern-transcription-service")
volume = Volume.from_name("transcription-volume", create_if_missing=True)

# Image Configuration
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.1.2",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "librosa==0.10.2",
        "accelerate==0.33.0",
        "ffmpeg-python",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "httpx>=0.24.0",
        "spacy==3.7.2",
    ])
    .apt_install(["ffmpeg"])
)

# FastAPI Web App with Custom Metadata
web_app = FastAPI(
    title="Modern Transcription Service",
    description="An advanced transcription service with real-time updates, PII detection, and WebSocket support.",
    version="2.0.0",
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.cls(
    image=image,
    gpu=gpu.T4(),
    timeout=1800,
    volumes={"/data": volume},
    secrets=[Secret.from_name("my-secure-secret")]
)
class WhisperTranscriptionService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.pipeline = None
        Path("/data").mkdir(parents=True, exist_ok=True)

    def detect_pii(self, text: str, words: List[Dict]) -> List[Dict]:
        """Detects PII (e.g., phone numbers, email addresses) in transcriptions."""
        pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "phone": r"\b\d{10,12}\b",
        }
        pii_matches = []
        for pii_type, pattern in pii_patterns.items():
            for match in re.finditer(pattern, text):
                pii_matches.append({"type": pii_type, "value": match.group(), "start": match.start(), "end": match.end()})
        return pii_matches

    async def transcribe_audio(
        self,
        audio_data: bytes,
        transcription_id: str,
        language: Optional[str] = "en",
    ) -> Dict:
        """Processes audio transcription with Whisper and PII detection."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                transcription = self.pipeline(tmp_file.name, language=language)
                pii_info = self.detect_pii(transcription["text"], transcription.get("words", []))
                result = {
                    "id": transcription_id,
                    "text": transcription["text"],
                    "pii_detected": bool(pii_info),
                    "pii_info": pii_info,
                }
                return result
        finally:
            os.unlink(tmp_file.name)


# Route Definitions
@web_app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    transcription_id: str = Form(...),
    language: Optional[str] = Form("en"),
):
    """API for initiating transcription."""
    service = WhisperTranscriptionService()
    audio_data = await audio.read()
    result = await service.transcribe_audio(audio_data, transcription_id, language)
    return JSONResponse(content=result)


@web_app.get("/transcription/{transcription_id}")
async def get_transcription(transcription_id: str):
    """API for fetching transcription."""
    # Example response
    return {"id": transcription_id, "status": "completed"}


@web_app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    await websocket.send_text("WebSocket connection established.")
    for progress in range(0, 101, 10):  # Simulated progress
        await asyncio.sleep(0.5)
        await websocket.send_json({"progress": progress})
    await websocket.send_text("Transcription complete.")
    await websocket.close()


@app.function(
    image=image,
    gpu=gpu.T4(),
    timeout=1800,
    volumes={"/data": volume},
)
@asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    app.run()

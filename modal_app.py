import os
import logging
import json
import hmac
import hashlib
import asyncio
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseSettings, HttpUrl, validator
from cryptography.fernet import Fernet
import httpx

# Configure logging without sensitive data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SecuritySettings(BaseSettings):
    """Application security settings"""
    MAX_VIDEO_DURATION: int = 300  # 5 minutes
    CLEANUP_INTERVAL: int = 86400  # 24 hours
    MAX_RETRIES: int = 3
    INITIAL_RETRY_DELAY: int = 1
    MIN_KEY_LENGTH: int = 32

    @validator('MAX_VIDEO_DURATION')
    def validate_max_duration(cls, v):
        if v <= 0 or v > 3600:  # Max 1 hour
            raise ValueError("Invalid video duration limit")
        return v

    class Config:
        env_prefix = "abc_"
        env_file = ".env"


settings = SecuritySettings()


class VideoEncryption:
    """Handles video encryption and decryption"""

    def __init__(self, key: Optional[bytes] = None):
        if not key or len(key) < settings.MIN_KEY_LENGTH:
            raise ValueError("Invalid encryption key length")

        try:
            self.key = key if isinstance(key, bytes) else key.encode()
            self.cipher_suite = Fernet(self.key)
        except Exception as e:
            logger.error("Encryption initialization failed")
            raise ValueError("Invalid encryption configuration")

    def encrypt_video(self, video_data: bytes) -> bytes:
        """Encrypts video data using Fernet"""
        try:
            return self.cipher_suite.encrypt(video_data)
        except Exception as e:
            logger.error("Video encryption failed")
            raise RuntimeError("Encryption error")

    def decrypt_video(self, encrypted_data: bytes) -> bytes:
        """Decrypts encrypted video data"""
        try:
            return self.cipher_suite.decrypt(encrypted_data)
        except Exception as e:
            logger.error("Video decryption failed")
            raise RuntimeError("Decryption error")


async def notify_webhook(
        callback_url: HttpUrl,
        result: Dict,
        app_secret: str
) -> bool:
    """
    Sends webhook notification with signed payload
    Returns: bool indicating success/failure
    """
    if not app_secret:
        logger.error("Missing webhook configuration")
        return False

    try:
        # Create signed payload
        payload = json.dumps(result, sort_keys=True)
        signature = hmac.new(
            app_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "X-Webhook-Signature": signature,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(settings.MAX_RETRIES):
                try:
                    response = await client.post(
                        str(callback_url),
                        json=result,
                        headers=headers,
                        timeout=10.0
                    )
                    if response.status_code < 400:
                        logger.info("Webhook notification successful")
                        return True

                    logger.warning(f"Webhook attempt {attempt + 1} failed")

                except Exception as e:
                    logger.error("Webhook request failed")
                    if attempt < settings.MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)

        return False

    except Exception as e:
        logger.error("Webhook notification failed")
        return False


app = FastAPI()


@app.get("/video/{video_id}")
async def get_video(video_id: str) -> StreamingResponse:
    """
    Retrieves and streams processed video

    Args:
        video_id: Unique video identifier

    Returns:
        StreamingResponse with video data

    Raises:
        HTTPException: For invalid requests or server errors
    """
    try:
        # Validate video_id
        if not video_id or not video_id.isalnum():
            raise HTTPException(status_code=400, detail="Invalid video ID")

        # Get video data (implement your video retrieval logic)
        video_data = await retrieve_video(video_id)

        if not video_data:
            raise HTTPException(status_code=404, detail="Video not found")

        # Stream video in chunks
        async def video_stream():
            chunk_size = 8192  # 8KB chunks
            for i in range(0, len(video_data), chunk_size):
                yield video_data[i:i + chunk_size]

        return StreamingResponse(
            video_stream(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=video_{video_id}.mp4",
                "Cache-Control": "no-store, must-revalidate",
                "Pragma": "no-cache",
            }
        )

    except Exception as e:
        logger.error(f"Error retrieving video: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Implement your video retrieval logic
async def retrieve_video(video_id: str) -> Optional[bytes]:
    """
    Implement your video retrieval logic here
    Returns video data or None if not found
    """
    pass
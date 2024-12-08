# Import statements and unchanged parts of the code remain the same
import os
from fastapi.responses import JSONResponse
from pathlib import Path

class VideoEncryption:
    def __init__(self, key: Optional[bytes] = None):
        if not key:
            raise ValueError("Encryption key is required")
        try:
            self.key = key if isinstance(key, bytes) else key.encode()
            self.cipher_suite = Fernet(self.key)
        except Exception as e:
            raise ValueError("Invalid encryption key") from e

    def encrypt_video(self, video_data: bytes) -> bytes:
        """Encrypts video data."""
        try:
            return self.cipher_suite.encrypt(video_data)
        except Exception as e:
            raise RuntimeError("Encryption failed") from e

    def decrypt_video(self, encrypted_data: bytes) -> bytes:
        """Decrypts encrypted video data."""
        try:
            return self.cipher_suite.decrypt(encrypted_data)
        except Exception as e:
            raise RuntimeError("Decryption failed") from e


# Updated Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Secure Environment Check for Webhook Secret
class Settings(BaseSettings):
    MAX_VIDEO_DURATION: int = 300  # 5 minutes max video duration
    CLEANUP_INTERVAL: int = 86400  # 24 hours
    MAX_RETRIES: int = 3
    INITIAL_RETRY_DELAY: int = 1

    class Config:
        env_prefix = "MODAL_"


settings = Settings()


# Webhook Notification with Minimal Data in Logs
async def notify_completion(callback_url: str, result: Dict):
    """Securely notifies the callback URL with the processing result."""
    if not callback_url or not callback_url.startswith(('http://', 'https://')):
        logger.error("Invalid callback URL provided")
        return

    secret = os.environ.get("WEBHOOK_SECRET")
    if not secret:
        logger.error("Webhook secret not configured")
        return

    try:
        serialized_payload = json.dumps(result, sort_keys=True, separators=(',', ':'))
        signature = hmac.new(secret.encode(), serialized_payload.encode(), hashlib.sha256).hexdigest()

        headers = {
            "X-Webhook-Signature": signature,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(settings.MAX_RETRIES):
                try:
                    response = await client.post(callback_url, json=result, headers=headers, timeout=10.0)
                    if response.status_code < 400:
                        logger.info("Webhook notification sent successfully")
                        return
                    logger.warning(f"Webhook attempt {attempt + 1} failed with status {response.status_code}")
                except Exception as e:
                    logger.error(f"Webhook attempt {attempt + 1} failed: {str(e)}")
                    if attempt < settings.MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
    except Exception as e:
        logger.error("Failed to send webhook notification")


# Get Video API Endpoint
@web_app.get("/video/{video_id}")
async def get_video(video_id: str):
    """Securely retrieves the encrypted video by ID."""
    model = EgoBlurModel()

    try:
        result = await model.get_video.remote.aio(video_id)

        if isinstance(result, JSONResponse):
            return result

        video_data, safe_video_id, file_size = result

        if not video_data:
            return JSONResponse(status_code=404, content={"error": "Video not found"})

        async def video_stream():
            chunk_size = 8192  # 8 KB chunks
            for i in range(0, len(video_data), chunk_size):
                yield video_data[i:i + chunk_size]

        return StreamingResponse(
            video_stream(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=processed_{safe_video_id}.mp4",
                "Content-Length": str(file_size),
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )
    except Exception as e:
        logger.error("Error retrieving video", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Server error"})

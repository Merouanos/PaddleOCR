
import os
import hmac
import tempfile
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from paddleocr import PaddleOCR

app = FastAPI(title="WinRahJibi OCR API")

API_KEY = os.environ.get("OCR_API_KEY", "")
OCR_LANG = os.environ.get("OCR_LANG", "fr")

# Load the OCR model once when the service starts.
ocr = PaddleOCR(lang=OCR_LANG)


def verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OCR_API_KEY is not configured",
        )

    expected = f"Bearer {API_KEY}"

    if not authorization or not hmac.compare_digest(
        authorization, expected
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ocr")
async def extract_text(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image type")

    contents = await file.read()

    if not contents or len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Invalid image size")

    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        results = ocr.predict(image)

        lines = []

        for result in results:
            texts = result["rec_texts"]
            scores = result["rec_scores"]

            for text, score in zip(texts, scores):
                lines.append({
                    "text": str(text),
                    "confidence": float(score),
                })

        return {"lines": lines}

    except HTTPException:
        raise
    except Exception:
        # Keep internal errors and receipt contents out of the response.
        raise HTTPException(status_code=500, detail="OCR processing failed")

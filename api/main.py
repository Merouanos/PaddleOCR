
import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("winrahjibi-ocr")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

API_KEY = os.environ.get("OCR_API_KEY", "")

# "fr" selects the Latin-language model family.
# "ar" selects the Arabic-language model family.
# Add other supported model configurations as needed.
OCR_LANGS = [
    lang.strip()
    for lang in os.environ.get("OCR_LANGS", "fr,ar").split(",")
    if lang.strip()
]

ocr_models: dict[str, PaddleOCR] = {}
ocr_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("OCR_API_KEY must be configured")

    if not OCR_LANGS:
        raise RuntimeError("Configure at least one OCR language model")

    logger.info("Loading %d OCR model(s)", len(OCR_LANGS))

    try:
        for lang in OCR_LANGS:
            logger.info("Loading OCR model configuration: %s", lang)

            model = await run_in_threadpool(
                lambda language=lang: PaddleOCR(
                    lang=language,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            )

            ocr_models[lang] = model

        logger.info("All OCR models loaded")
        yield

    except Exception:
        logger.exception("OCR service startup or runtime failure")
        raise
    finally:
        ocr_models.clear()


app = FastAPI(
    title="WinRahJibi OCR API",
    lifespan=lifespan,
)


def verify_api_key(authorization: str | None) -> None:
    if not authorization or not hmac.compare_digest(
        authorization,
        f"Bearer {API_KEY}",
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def read_limited_upload(file: UploadFile) -> bytes:
    chunks = []
    total = 0

    while True:
        chunk = await file.read(64 * 1024)

        if not chunk:
            break

        total += len(chunk)

        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit",
            )

        chunks.append(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="Empty image")

    return b"".join(chunks)


def polygon_bounds(polygon):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

    if len(points) < 4:
        return None

    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())

    return x1, y1, x2, y2


def intersection_over_union(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def contains_arabic(text: str) -> bool:
    return any(
        "\u0600" <= char <= "\u06ff"
        or "\u0750" <= char <= "\u077f"
        or "\u08a0" <= char <= "\u08ff"
        for char in text
    )


def collect_lines(results, model_lang: str) -> list[dict]:
    lines = []

    for result in results or []:
        try:
            texts = result["rec_texts"]
            scores = result["rec_scores"]
            polygons = result["rec_polys"]
        except (KeyError, TypeError, AttributeError):
            continue

        for text, score, polygon in zip(texts, scores, polygons):
            text = str(text).strip()

            if not text:
                continue

            bounds = polygon_bounds(polygon)

            if bounds is None:
                continue

            x1, y1, x2, y2 = bounds

            lines.append({
                "text": text,
                "confidence": float(score),
                "language_model": model_lang,
                "box": {
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                },
                "_bounds": bounds,
            })

    return lines


def merge_lines(lines: list[dict]) -> list[dict]:
    # Each model detects the same receipt regions. Remove highly
    # overlapping duplicate lines, retaining the higher-confidence result.
    lines.sort(key=lambda line: line["confidence"], reverse=True)

    merged = []

    for candidate in lines:
        duplicate = any(
            intersection_over_union(
                candidate["_bounds"],
                existing["_bounds"],
            ) >= 0.5
            for existing in merged
        )

        if not duplicate:
            merged.append(candidate)

    # Approximate reading order: top to bottom, then left to right.
    # Arabic lines are ordered right to left.
    merged.sort(
        key=lambda line: (
            round(line["_bounds"][1] / 10),
            -line["_bounds"][0]
            if contains_arabic(line["text"])
            else line["_bounds"][0],
        )
    )

    for line in merged:
        line.pop("_bounds", None)

    return merged


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(ocr_models.keys()),
    }


@app.post("/v1/ocr")
async def extract_text(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and WebP images are supported",
        )

    contents = await read_limited_upload(file)

    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    height, width = image.shape[:2]

    if height * width > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions are too large",
        )

    try:
        all_lines = []

        # Run models sequentially to limit peak resource usage.
        async with ocr_lock:
            for lang, model in ocr_models.items():
                results = await run_in_threadpool(model.predict, image)
                all_lines.extend(collect_lines(results, lang))

        lines = merge_lines(all_lines)

        return {
            "lines": lines,
            "models_used": list(ocr_models.keys()),
            "line_count": len(lines),
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR inference failed")
        raise HTTPException(
            status_code=500,
            detail="OCR processing failed",
        )
    finally:
        await file.close()

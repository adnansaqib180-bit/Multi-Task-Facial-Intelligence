from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_DIR = Path(__file__).resolve().parents[1] / "trained models"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
EMOTION_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import keras

    model_files = {
        "age_gender": "age_gender.keras",
        "emotion": "emotion_model.keras",
        "real_vs_ai": "real_vs_ai.keras",
    }
    app.state.models = {
        name: keras.models.load_model(MODEL_DIR / filename)
        for name, filename in model_files.items()
    }
    yield


app = FastAPI(
    title="Multi-Task Facial Intelligence API",
    description="Run age and gender, facial emotion, and real-versus-AI image predictions.",
    version="1.0.0",
    lifespan=lifespan,
)


def _read_image(image: UploadFile) -> Image.Image:
    content = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit.")

    try:
        source = Image.open(BytesIO(content))
        if source.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG, PNG, and WebP images are supported.",
            )
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions are too large.")
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        ) from error


def _model_input(image: Image.Image, size: int, grayscale: bool = False) -> np.ndarray:
    if grayscale:
        image = ImageOps.grayscale(image)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def _predict(request: Request, model_name: str, inputs: np.ndarray):
    return request.app.state.models[model_name].predict(inputs, verbose=0)


@app.post("/predict/age-gender")
def predict_age_gender(
    request: Request,
    image: Annotated[UploadFile, File(description="A face image (JPEG, PNG, or WebP).")],
):
    face = _read_image(image)
    prediction = _predict(request, "age_gender", _model_input(face, 224))
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 2:
        raise RuntimeError("The age/gender model returned an unexpected output shape.")

    age = float(np.asarray(prediction[0]).reshape(-1)[0])
    female_probability = float(np.asarray(prediction[1]).reshape(-1)[0])
    return {
        "age": age,
        "gender": "female" if female_probability >= 0.5 else "male",
        "gender_probabilities": {
            "female": female_probability,
            "male": 1.0 - female_probability,
        },
    }


@app.post("/predict/emotion")
def predict_emotion(
    request: Request,
    image: Annotated[UploadFile, File(description="A face image (JPEG, PNG, or WebP).")],
):
    face = _read_image(image)
    scores = np.asarray(
        _predict(request, "emotion", _model_input(face, 124, grayscale=True))
    ).reshape(-1)
    if scores.size != len(EMOTION_LABELS):
        raise RuntimeError("The emotion model returned an unexpected number of classes.")

    class_index = int(np.argmax(scores))
    return {
        "emotion": EMOTION_LABELS[class_index],
        "confidence": float(scores[class_index]),
        "probabilities": {
            label: float(score) for label, score in zip(EMOTION_LABELS, scores)
        },
    }


@app.post("/predict/real-vs-ai")
def predict_real_vs_ai(
    request: Request,
    image: Annotated[UploadFile, File(description="An image (JPEG, PNG, or WebP).")],
):
    picture = _read_image(image)
    prediction = np.asarray(
        _predict(request, "real_vs_ai", _model_input(picture, 224))
    ).reshape(-1)
    if prediction.size != 1:
        raise RuntimeError("The real-versus-AI model returned an unexpected output shape.")

    real_probability = float(prediction[0])
    return {
        "label": "real" if real_probability >= 0.5 else "ai-generated",
        "confidence": max(real_probability, 1.0 - real_probability),
        "probabilities": {
            "real": real_probability,
            "ai-generated": 1.0 - real_probability,
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}

from pathlib import Path
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "ui.html")
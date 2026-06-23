import os
import csv
import gc
import time
import logging
import urllib.request
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

SELF_PING_INTERVAL = int(os.getenv("SELF_PING_INTERVAL", 9000))  # 150 minutes in seconds
SELF_PING_URL = os.getenv("SELF_PING_URL", "")

logger = logging.getLogger(__name__)


def _self_ping_worker():
    """Background worker that pings the server's own health endpoint
    every SELF_PING_INTERVAL seconds to keep the Hugging Face Space awake."""
    while True:
        time.sleep(SELF_PING_INTERVAL)
        ping_url = SELF_PING_URL or os.getenv("SELF_PING_URL", "")
        if not ping_url:
            logger.warning("[keep-alive] SELF_PING_URL not set, skipping ping")
            continue
        try:
            req = urllib.request.Request(ping_url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                logger.info(
                    "[keep-alive] pinged %s — status %s", ping_url, resp.status
                )
        except Exception as exc:
            logger.warning("[keep-alive] ping failed: %s", exc)


def start_keep_alive():
    """Spawn the keep-alive daemon thread."""
    ping_url = SELF_PING_URL or os.getenv("SELF_PING_URL", "")
    if not ping_url:
        logger.info(
            "[keep-alive] SELF_PING_URL not configured — set it to enable self-ping "
            "(e.g. https://<your-space>.hf.space/api/health)"
        )
        return
    thread = Thread(target=_self_ping_worker, daemon=True, name="keep-alive-ping")
    thread.start()
    logger.info(
        "[keep-alive] started — pinging %s every %d seconds (~%d minutes)",
        ping_url,
        SELF_PING_INTERVAL,
        SELF_PING_INTERVAL // 60,
    )

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from model import CNN


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DISEASE_INFO_PATH = Path(os.getenv("DISEASE_INFO_PATH", DATA_DIR / "disease_info.csv"))
SUPPLEMENT_INFO_PATH = Path(os.getenv("SUPPLEMENT_INFO_PATH", DATA_DIR / "supplement_info.csv"))
CLASS_COUNT = 39
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "plant_disease_model_1_latest.pt")
MODEL_LOCK = Lock()
MODEL_STATE_LOCK = Lock()
MODEL_STATE = {"status": "idle", "error": ""}
MODEL_WARMUP_THREAD = None

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def load_csv_data():
    if not DISEASE_INFO_PATH.exists():
        raise FileNotFoundError(f"Disease info CSV not found at {DISEASE_INFO_PATH}")

    if not SUPPLEMENT_INFO_PATH.exists():
        raise FileNotFoundError(f"Supplement info CSV not found at {SUPPLEMENT_INFO_PATH}")

    disease_rows = None
    supplement_rows = None
    encodings = ("utf-8-sig", "utf-8", "cp1252")

    for encoding in encodings:
        try:
            with DISEASE_INFO_PATH.open("r", encoding=encoding, newline="") as disease_file:
                disease_rows = list(csv.DictReader(disease_file))
            break
        except UnicodeDecodeError:
            continue

    for encoding in encodings:
        try:
            with SUPPLEMENT_INFO_PATH.open("r", encoding=encoding, newline="") as supplement_file:
                supplement_rows = list(csv.DictReader(supplement_file))
            break
        except UnicodeDecodeError:
            continue

    if disease_rows is None:
        raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {DISEASE_INFO_PATH.name}")

    if supplement_rows is None:
        raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {SUPPLEMENT_INFO_PATH.name}")

    if not disease_rows:
        raise ValueError("Disease info CSV is empty")

    if not supplement_rows:
        raise ValueError("Supplement info CSV is empty")

    return disease_rows, supplement_rows


@lru_cache(maxsize=1)
def get_catalog_data():
    return load_csv_data()


def resolve_model_path() -> Path:
    local_model_path = os.getenv("MODEL_PATH")
    if local_model_path:
        resolved_path = Path(local_model_path).expanduser().resolve()
        if resolved_path.exists():
            return resolved_path

    repo_id = os.getenv("HF_MODEL_REPO_ID")
    if not repo_id:
        raise RuntimeError(
            "Set HF_MODEL_REPO_ID for Hugging Face model loading or MODEL_PATH for a local model file."
        )

    download_path = hf_hub_download(
        repo_id=repo_id,
        filename=MODEL_FILENAME,
        token=os.getenv("HF_TOKEN") or None,
        local_dir=BASE_DIR / ".cache" / "hf-models",
    )
    return Path(download_path)


def load_model():
    checkpoint = torch.load(resolve_model_path(), map_location="cpu")
    model = CNN(CLASS_COUNT)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    del checkpoint
    gc.collect()
    return model


@lru_cache(maxsize=1)
def get_model():
    with MODEL_LOCK:
        return load_model()


def get_model_state():
    with MODEL_STATE_LOCK:
        return dict(MODEL_STATE)


def get_catalog_state():
    try:
        disease_rows, supplement_rows = get_catalog_data()
    except Exception as error:
        return {"loaded": False, "error": str(error), "diseaseRows": 0, "supplementRows": 0}

    return {
        "loaded": True,
        "error": "",
        "diseaseRows": len(disease_rows),
        "supplementRows": len(supplement_rows),
    }


def set_model_state(status: str, error: str = ""):
    with MODEL_STATE_LOCK:
        MODEL_STATE["status"] = status
        MODEL_STATE["error"] = error


def _load_model_in_background():
    global MODEL_WARMUP_THREAD

    try:
        get_model()
    except Exception as error:
        set_model_state("error", str(error))
    else:
        set_model_state("ready")
    finally:
        with MODEL_STATE_LOCK:
            MODEL_WARMUP_THREAD = None


def ensure_model_warmup(force: bool = False):
    global MODEL_WARMUP_THREAD

    with MODEL_STATE_LOCK:
        current_state = dict(MODEL_STATE)
        has_active_thread = MODEL_WARMUP_THREAD is not None and MODEL_WARMUP_THREAD.is_alive()

        if current_state["status"] == "ready":
            return current_state

        if has_active_thread and not force:
            return current_state

        MODEL_STATE["status"] = "loading"
        MODEL_STATE["error"] = ""
        MODEL_WARMUP_THREAD = Thread(target=_load_model_in_background, daemon=True)
        MODEL_WARMUP_THREAD.start()
        return dict(MODEL_STATE)


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    resized_image = image.resize((224, 224))
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))
    return image_tensor.unsqueeze(0)


def format_label(raw_label: str) -> str:
    return raw_label.replace("___", " - ").replace("_", " ").replace(",", "")


def is_healthy_label(raw_label: str) -> bool:
    return "healthy" in raw_label.lower()


def build_prediction_response(prediction_index: int, confidence: float):
    disease_info, supplement_info = get_catalog_data()
    disease_row = disease_info[prediction_index]
    supplement_row = supplement_info[prediction_index]
    raw_name = normalize_text(supplement_row["disease_name"])
    disease_name = normalize_text(disease_row["disease_name"])

    return {
        "predictionIndex": prediction_index,
        "className": raw_name,
        "title": disease_name,
        "displayName": format_label(raw_name or disease_name),
        "confidence": round(confidence * 100, 2),
        "isHealthy": is_healthy_label(raw_name or disease_name),
        "description": normalize_text(disease_row["description"]),
        "possibleSteps": normalize_text(disease_row["Possible Steps"]),
        "referenceImage": normalize_text(disease_row["image_url"]),
        "supplement": {
            "name": normalize_text(supplement_row["supplement name"]),
            "image": normalize_text(supplement_row["supplement image"]),
            "buyLink": normalize_text(supplement_row["buy link"]),
        },
    }


# Maximum upload size: 10 MB
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_keep_alive()
    yield


def create_app() -> FastAPI:
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    application = FastAPI(
        title="Plant Disease Detection API",
        description="AI-powered plant disease detection using PyTorch CNN model",
        version="2.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Root ──────────────────────────────────────────────────────────
    @application.get("/")
    async def index():
        return {
            "name": "Plant Disease Detection API",
            "status": "ok",
            "message": "Use the /api endpoints from the frontend or API client.",
            "endpoints": {
                "health": "/api/health",
                "catalog": "/api/catalog",
                "warmup": "/api/warmup",
                "predict": "/api/predict",
            },
        }

    # ── Health ────────────────────────────────────────────────────────
    @application.get("/api/health")
    async def health_check():
        model_state = get_model_state()
        catalog_state = get_catalog_state()
        return {
            "status": "ok",
            "model": MODEL_FILENAME,
            "classes": CLASS_COUNT,
            "modelLoaded": model_state["status"] == "ready",
            "modelStatus": model_state["status"],
            "modelError": model_state["error"],
            "catalogLoaded": catalog_state["loaded"],
            "catalogError": catalog_state["error"],
            "catalogRows": {
                "diseaseInfo": catalog_state["diseaseRows"],
                "supplementInfo": catalog_state["supplementRows"],
            },
        }

    # ── Warmup ────────────────────────────────────────────────────────
    @application.post("/api/warmup")
    async def warmup_model():
        model_state = get_model_state()
        if model_state["status"] == "ready":
            return {"status": "ok", "modelLoaded": True, "modelStatus": "ready"}

        model_state = ensure_model_warmup(force=model_state["status"] == "error")
        return JSONResponse(
            status_code=202,
            content={
                "status": "warming",
                "modelLoaded": False,
                "modelStatus": model_state["status"],
                "message": "Model warmup started. Retry prediction after the backend finishes loading the model.",
            },
        )

    # ── Catalog ───────────────────────────────────────────────────────
    @application.get("/api/catalog")
    async def catalog():
        try:
            _, supplement_info = get_catalog_data()
        except Exception as error:
            return JSONResponse(
                status_code=500,
                content={"error": f"Catalog data unavailable: {error}"},
            )

        crop_names = sorted(
            {
                format_label(label.split(" - ")[0])
                for label in (format_label(row.get("disease_name", "")) for row in supplement_info)
                if label and "background without leaves" not in label.lower()
            }
        )
        return {
            "projectName": "Plant Disease Detection",
            "supportedCrops": crop_names,
            "totalClasses": CLASS_COUNT,
        }

    # ── Predict ───────────────────────────────────────────────────────
    @application.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        model_state = get_model_state()
        if model_state["status"] != "ready":
            model_state = ensure_model_warmup(force=model_state["status"] == "error")
            if model_state["status"] != "ready":
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Model is still warming up. Wait a moment and try again.",
                        "modelLoaded": False,
                        "modelStatus": model_state["status"],
                    },
                    headers={"Retry-After": "20"},
                )

        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"error": "Please choose an image before submitting."},
            )

        # Read file content with size check
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "Image is too large. Use an image smaller than 10 MB."},
            )

        try:
            import io
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
        except (UnidentifiedImageError, Exception):
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported image format. Upload PNG, JPG, or JPEG."},
            )

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            try:
                model = get_model()
            except Exception as error:
                set_model_state("error", str(error))
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Model could not be loaded: {error}"},
                )

            output_tensor = model(input_tensor)
            probabilities = F.softmax(output_tensor, dim=1)[0]
            prediction_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[prediction_index].item())

        try:
            response_payload = build_prediction_response(prediction_index, confidence)
        except Exception as error:
            return JSONResponse(
                status_code=500,
                content={"error": f"Prediction metadata unavailable: {error}"},
            )

        return response_payload

    # ── Exception handlers ────────────────────────────────────────────
    @application.exception_handler(413)
    async def payload_too_large(request: Request, exc):
        return JSONResponse(
            status_code=413,
            content={"error": "Image is too large. Use an image smaller than 10 MB."},
        )

    @application.exception_handler(500)
    async def internal_server_error(request: Request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {exc}"},
        )

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        reload=True,
        log_level="info",
    )

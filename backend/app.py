import os
import csv
import gc
from functools import lru_cache
from pathlib import Path
from threading import Lock

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
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

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def load_csv_data():
    with DISEASE_INFO_PATH.open("r", encoding="cp1252", newline="") as disease_file:
        disease_rows = list(csv.DictReader(disease_file))

    with SUPPLEMENT_INFO_PATH.open("r", encoding="cp1252", newline="") as supplement_file:
        supplement_rows = list(csv.DictReader(supplement_file))

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


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    allowed_origins = [
        origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if origin.strip()
    ]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins or "*"}})

    @app.get("/api/health")
    def health_check():
        return jsonify(
            {
                "status": "ok",
                "model": MODEL_FILENAME,
                "classes": CLASS_COUNT,
                "modelLoaded": get_model.cache_info().currsize > 0,
            }
        )

    @app.post("/api/warmup")
    def warmup_model():
        try:
            get_model()
        except Exception as error:
            return jsonify({"status": "error", "error": f"Model warmup failed: {error}"}), 500

        return jsonify({"status": "ok", "modelLoaded": True})

    @app.get("/api/catalog")
    def catalog():
        _, supplement_info = get_catalog_data()
        crop_names = sorted(
            {
                format_label(label.split(" - ")[0])
                for label in (format_label(row.get("disease_name", "")) for row in supplement_info)
                if label and "background without leaves" not in label.lower()
            }
        )
        return jsonify(
            {
                "projectName": "Plant Disease Detection",
                "supportedCrops": crop_names,
                "totalClasses": CLASS_COUNT,
            }
        )

    @app.post("/api/predict")
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "Image file is required under the 'file' field."}), 400

        image_file = request.files["file"]
        if not image_file.filename:
            return jsonify({"error": "Please choose an image before submitting."}), 400

        try:
            image = Image.open(image_file.stream).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Unsupported image format. Upload PNG, JPG, or JPEG."}), 400

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            try:
                model = get_model()
            except Exception as error:
                return jsonify({"error": f"Model could not be loaded: {error}"}), 500

            output_tensor = model(input_tensor)
            probabilities = F.softmax(output_tensor, dim=1)[0]
            prediction_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[prediction_index].item())

        return jsonify(build_prediction_response(prediction_index, confidence))

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"error": "Image is too large. Use an image smaller than 10 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

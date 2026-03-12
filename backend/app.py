import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import CNN


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DISEASE_INFO_PATH = Path(os.getenv("DISEASE_INFO_PATH", DATA_DIR / "disease_info.csv"))
SUPPLEMENT_INFO_PATH = Path(os.getenv("SUPPLEMENT_INFO_PATH", DATA_DIR / "supplement_info.csv"))
CLASS_COUNT = 39
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "plant_disease_model_1_latest.pt")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_csv_data():
    disease_frame = pd.read_csv(DISEASE_INFO_PATH, encoding="cp1252")
    supplement_frame = pd.read_csv(SUPPLEMENT_INFO_PATH, encoding="cp1252")
    return disease_frame, supplement_frame


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
    return model


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def format_label(raw_label: str) -> str:
    return raw_label.replace("___", " - ").replace("_", " ").replace(",", "")


def is_healthy_label(raw_label: str) -> bool:
    return "healthy" in raw_label.lower()


def build_prediction_response(prediction_index: int, confidence: float):
    disease_row = DISEASE_INFO.iloc[prediction_index]
    supplement_row = SUPPLEMENT_INFO.iloc[prediction_index]
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
        return jsonify({"status": "ok", "model": MODEL_FILENAME, "classes": CLASS_COUNT})

    @app.get("/api/catalog")
    def catalog():
        crop_names = sorted(
            {
                format_label(label.split(" - ")[0])
                for label in (format_label(name) for name in SUPPLEMENT_INFO["disease_name"].fillna(""))
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

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output_tensor = MODEL(input_tensor)
            probabilities = F.softmax(output_tensor, dim=1)[0]
            prediction_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[prediction_index].item())

        return jsonify(build_prediction_response(prediction_index, confidence))

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"error": "Image is too large. Use an image smaller than 10 MB."}), 413

    return app


DISEASE_INFO, SUPPLEMENT_INFO = load_csv_data()
MODEL = load_model()
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

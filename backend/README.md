---
title: Plant Disease Detection API
sdk: docker
app_port: 7860
pinned: false
---

This folder can be deployed as a Hugging Face Docker Space without changing the frontend API contract.

## Exposed Endpoints

- `GET /api/health`
- `GET /api/catalog`
- `POST /api/warmup`
- `POST /api/predict`

## Space Setup

1. Create a new Hugging Face Space.
2. Choose `Docker` as the Space SDK.
3. Upload the contents of this `backend/` folder as the Space repository root.
4. Add these Space variables:

```text
CORS_ALLOWED_ORIGINS=https://save-plants.vercel.app
HF_MODEL_REPO_ID=your-huggingface-username/plant-disease-detection-model
HF_MODEL_FILENAME=plant_disease_model_1_latest.pt
```

1. If the model repository is private, add this Space secret:

```text
HF_TOKEN=your_hugging_face_token
```

1. Wait for the build to finish.
1. After the Space starts, call `POST /api/warmup` once.

## Frontend Configuration

Point the frontend at the Space URL:

```text
VITE_API_BASE_URL=https://your-space-name.hf.space
```

The frontend does not need any route changes because the Space serves the same `/api/*` endpoints as the Render backend.

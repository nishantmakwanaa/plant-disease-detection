# Plant Disease Detection

Plant Disease Detection is a modernized split-deployment Flask + PyTorch project. The app now targets a free production setup with:

- `frontend/` for a React + Vite UI deployed on Vercel
- `backend/` for a Flask API deployed on Render
- a PyTorch model stored on Hugging Face and downloaded at backend startup

If Render free-tier memory is not enough for PyTorch inference, the same backend can also be deployed as a Hugging Face Docker Space.

## Project Details

- Project name: Plant Disease Detection
- Created by: Nishant Makwana
- GitHub: [github.com/nishantmakwanaa](https://github.com/nishantmakwanaa)
- LinkedIn: [linkedin.com/in/nishantmakwanaa](https://linkedin.com/in/nishantmakwanaa)
- Portfolio: [nishantmakwanaa.lovable.app](https://nishantmakwanaa.lovable.app)

## Architecture

```text
User
    -> Frontend (React UI on Vercel)
    -> Backend API (Flask on Render)
    -> Model weights (PyTorch file on Hugging Face)
```

## Folder Layout

```text
frontend/
    React UI for Vercel

backend/
    Flask API for Render
    data/
    Procfile
    render.yaml
    requirements.txt

Model/
    Original notebook assets
```

## Local Development

### 1. Start the backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Set either of these before starting the backend:

- `HF_MODEL_REPO_ID` to download the model from Hugging Face
- or `MODEL_PATH` to point to a local `.pt` file during development

### 2. Start the frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Set `VITE_API_BASE_URL=http://localhost:5000` for local development.

## Deploy Frontend on Vercel

1. Push the repository to GitHub.
2. Import the repo into Vercel.
3. Set the Vercel root directory to `frontend`.
4. Add the environment variable:

```text
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

1. Deploy.

## Deploy Backend on Render

1. Create a new Web Service in Render.
2. Connect the GitHub repository.
3. Set root directory to `backend`.
4. Pin Python to `3.11.8`.
5. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180
```

1. Add these environment variables in Render:

```text
PYTHON_VERSION=3.11.8
HF_MODEL_REPO_ID=your-huggingface-username/plant-disease-detection-model
HF_MODEL_FILENAME=plant_disease_model_1_latest.pt
CORS_ALLOWED_ORIGINS=https://your-vercel-project.vercel.app
```

If you deploy with `render.yaml`, the repository already pins Python for you. This matters because newer Render defaults, such as Python 3.14, do not have compatible wheels for the pinned `pandas` and `torch` versions in this backend.

The backend now starts model warmup asynchronously when you call `POST /api/warmup`, or when the first `POST /api/predict` request arrives and the model is still idle. The warmup endpoint returns immediately with `202 Accepted` while the model downloads in the background.

While the model is still loading, `POST /api/predict` returns `503` with a JSON error payload instead of hanging until the hosting platform returns `502`.

If you want to bypass Hugging Face during local testing, set `MODEL_PATH` instead.

## Deploy Backend on Hugging Face Spaces

If Render keeps failing with `Ran out of memory (used over 512MB)`, move the backend to a Hugging Face Docker Space. That gives you a larger free runtime and avoids the Render free-tier memory cap.

1. Create a new Space on Hugging Face.
2. Choose `Docker` as the SDK.
3. Copy the contents of [backend/README.md](backend/README.md), [backend/Dockerfile](backend/Dockerfile), and the rest of the `backend/` folder into the Space repository.
4. Add these Space variables:

```text
CORS_ALLOWED_ORIGINS=https://save-plants.vercel.app
HF_MODEL_REPO_ID=your-huggingface-username/plant-disease-detection-model
HF_MODEL_FILENAME=plant_disease_model_1_latest.pt
```

1. If your model repo is private, add `HF_TOKEN` as a Space secret.
1. Wait for the image build to complete.
1. Set the frontend environment variable to the Space URL:

```text
VITE_API_BASE_URL=https://your-space-name.hf.space
```

1. Call `POST /api/warmup` once after the Space is live.

The frontend already uses the same `/api/catalog` and `/api/predict` paths, so only the base URL needs to change.

## Host the Model on Hugging Face

1. Create a new model repository on Hugging Face.
2. Upload your trained model file, for example `plant_disease_model_1_latest.pt`.
3. If the repository is private, create a Hugging Face access token and set `HF_TOKEN` in Render.
4. Put the repository name into `HF_MODEL_REPO_ID`.

The backend is already configured to call `hf_hub_download(...)` automatically.

## API Endpoints

- `GET /api/health` returns API status
- `GET /api/catalog` returns supported crops and class count
- `POST /api/predict` accepts multipart form data with a `file` field

Example frontend request:

```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
});
```

## What Changed

- Reworked the UI into a responsive React experience with a new earthy color palette and motion.
- Replaced server-rendered HTML flow with a JSON API contract for deployment flexibility.
- Moved model delivery out of the repository and into Hugging Face compatible loading.
- Moved disease metadata into `backend/data/` so the deployed API is self-contained.
- Updated project branding and creator details.

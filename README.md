# Plant Disease Detection

Plant Disease Detection is a modernized split-deployment Flask + PyTorch project. The app now targets a free production setup with:

- `frontend/` for a React + Vite UI deployed on Vercel
- `backend/` for a Flask API deployed on Render
- a PyTorch model stored on Hugging Face and downloaded at backend startup

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
4. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

1. Add these environment variables in Render:

```text
HF_MODEL_REPO_ID=your-huggingface-username/plant-disease-detection-model
HF_MODEL_FILENAME=plant_disease_model_1_latest.pt
CORS_ALLOWED_ORIGINS=https://your-vercel-project.vercel.app
```

If you want to bypass Hugging Face during local testing, set `MODEL_PATH` instead.

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

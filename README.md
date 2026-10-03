<div align="center">

# ✨ Multi-Task Facial Intelligence

### One image. Three vision models. A beautiful, interactive experience.

Analyze age and gender, classify facial emotion, and check whether an image is
real or AI-generated — all through a lightweight FastAPI service and a polished
web interface.

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow and Keras">
</p>

[Quick start](#-quick-start) · [Features](#-features) · [API](#-api) · [Project layout](#-project-layout)

</div>

---

## 🎬 Screen recording

[▶ Watch the project screen recording]([video/Screen%20Recording%202026-10-03%20190016.mp4](https://github.com/user-attachments/assets/c52161c5-5d0e-4fc1-ae04-a0fa623303fb))



## 🌟 Features

- **Age & gender estimation** — returns an estimated age and gender probabilities.
- **Emotion classification** — predicts angry, disgust, fear, happy, neutral,
  sad, or surprise, with confidence scores.
- **Real vs. AI-generated classification** — returns probabilities for both
  classes.
- **Interactive browser UI** — upload or drag-and-drop an image, choose a task,
  and view results.
- **FastAPI documentation** — explore and test the endpoints at `/docs`.
- **Input checks** — accepts JPEG, PNG, and WebP images up to 10 MiB.

## 🚀 Quick start

### 1. Clone the repository

```bash
git clone https://github.com/adnansaqib180-bit/Multi-Task-Facial-Intelligence.git
cd Multi-Task-Facial-Intelligence
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the trained models

The model weights are not tracked in this repository, so a fresh clone does not
include them. Obtain the model files from the project maintainer and place them
in the `trained models/` folder with these exact filenames:

```text
trained models/
├── age_gender.keras
├── emotion_model.keras
└── real_vs_ai.keras
```

The API loads all three models when it starts. The server will not start
successfully until these files are available.

### 5. Start the Uvicorn server

From the repository root, with the virtual environment activated:

```bash
python -m uvicorn api.main:app --reload
```

Then open:

- **Web app:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive API docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Press **Ctrl+C** in the terminal to stop the server.

## 🧠 API

Each prediction endpoint accepts an uploaded image in the multipart form field
named `image`.

| Method | Endpoint | Result |
|---|---|---|
| `POST` | `/predict/age-gender` | Estimated age, gender, and gender probabilities |
| `POST` | `/predict/emotion` | Predicted emotion and class probabilities |
| `POST` | `/predict/real-vs-ai` | `real` or `ai-generated` label and probabilities |
| `GET` | `/health` | API health status |

Example request using `curl`:

```bash
curl -X POST \
  -F "image=@face.jpg" \
  http://127.0.0.1:8000/predict/emotion
```

On Windows PowerShell, the same request can be sent with:

```powershell
curl.exe -X POST -F "image=@face.jpg" http://127.0.0.1:8000/predict/emotion
```

## 🗂️ Project layout

```text
Multi-Task-Facial-Intelligence/
├── api/
│   ├── main.py             # FastAPI app and prediction endpoints
│   └── ui.html             # Browser-based interface
├── trained models/         # Locally supplied .keras model files
├── training and testing/   # Model training scripts
├── video/                  # Local screen recording (Git-ignored)
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚠️ Responsible use

Predictions are model estimates and can be inaccurate or biased. Do not use
them as the sole basis for consequential decisions about people.

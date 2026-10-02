# Multi-Task-Facial-Intelligence

## API

Install the dependencies and start the API from the project root:

```bash
python -m pip install -r requirments.txt
uvicorn api.main:app --reload
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.
Each prediction endpoint accepts an image as multipart form data in the `image`
field:

- `POST /predict/age-gender` returns an estimated age, predicted gender, and
  gender probabilities.
- `POST /predict/emotion` returns the predicted emotion, confidence, and
  probabilities for angry, disgust, fear, happy, neutral, sad, and surprise.
- `POST /predict/real-vs-ai` returns `real` or `ai-generated`, confidence, and
  probabilities for both classes.
- `GET /health` checks that the API process is running.

For example:

```bash
curl -X POST -F "image=@face.jpg" http://127.0.0.1:8000/predict/emotion
```
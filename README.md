# Iris ML Pipeline: MLflow, FastAPI and Docker

End-to-end MLOps pipeline: train a Random Forest on the Iris dataset, track experiments with MLflow, serve predictions with FastAPI and run it in Docker.

## Run locally
```bash
pip install -r requirements.txt
python src/train.py --n_estimators 100 --max_depth 5
mlflow ui --workers 1
python -m uvicorn app.main:app --reload
```

## Run with Docker
```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```
Open http://localhost:8000/docs

## Author
Your Name | Roll No. | Course
Rohan ADT24SOCBD213 MLOPS


# ✈️ Flight Delay Prediction — End-to-End ML System

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/flight-delay-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/flight-delay-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-orange.svg)](https://mlflow.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-yellow.svg)](https://huggingface.co/spaces)

> **A production-grade ML system** for predicting flight delays — covering ETL, ML modeling, MLOps, FastAPI, Docker, CI/CD, and cloud deployment.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/flight-delay-predictor)

---

## 📋 Project Objective

Build a **Flight Delay Prediction Platform** that:

1. 📥 Extracts raw aviation data from BTS Open Data
2. 🔄 Runs a structured ETL pipeline
3. 🗄️ Stores curated data in SQLite
4. 🤖 Trains ML models (XGBoost, LightGBM, Random Forest)
5. 📊 Tracks experiments with MLflow
6. 📦 Packages & deploys the best model
7. ☁️ Exposes a cloud inference endpoint (Hugging Face Spaces)
8. 🔍 Implements monitoring & basic MLOps lifecycle

---

## 🗂️ Project Structure

```
flight-delay-prediction/
├── .github/workflows/ci.yml     # CI/CD pipeline
├── data/
│   ├── raw/                     # Raw downloaded data
│   └── processed/               # Cleaned feature data
├── database/flight_delay.db     # SQLite database
├── etl/
│   ├── extract.py               # Download/generate data
│   ├── transform.py             # Clean + feature engineer
│   └── load.py                  # Store to database
├── src/
│   ├── train.py                 # Model training + MLflow
│   ├── evaluate.py              # Metrics + plots
│   └── predict.py               # Inference helper
├── api/
│   ├── main.py                  # FastAPI app
│   └── schemas.py               # Pydantic models
├── app.py                       # Hugging Face Gradio UI
├── monitoring/drift_detection.py
├── tests/
│   ├── test_etl.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/flight-delay-prediction.git
cd flight-delay-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Step 1: Extract data (generates sample data if Kaggle not configured)
python etl/extract.py

# Step 2: Transform & feature engineering
python etl/transform.py

# Step 3: Load to database
python etl/load.py

# Step 4: Train models (with MLflow tracking)
python src/train.py

# Step 5: Evaluate best model
python src/evaluate.py
```

### 3. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Visit: http://localhost:8000/docs
```

### 4. Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Flight delay prediction |
| `GET` | `/model-info` | Model details |
| `GET` | `/docs` | Swagger UI |

### Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "month": 6,
    "day_of_week": 5,
    "day_of_month": 15,
    "dep_hour": 8,
    "carrier": "AA",
    "origin": "JFK",
    "dest": "LAX",
    "distance": 2475.0,
    "crs_elapsed_time": 330.0
  }'
```

### Sample Response

```json
{
  "delayed": 1,
  "probability": 0.7312,
  "status": "DELAYED",
  "confidence": "HIGH"
}
```

---

## 🐳 Docker

```bash
# Build & run locally
docker build -t flight-delay-api .
docker run -p 8000:8000 -v ./models:/app/models flight-delay-api

# Or use docker-compose (API + MLflow server)
docker-compose up --build
```

---

## 📊 MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow server --host 0.0.0.0 --port 5000

# Visit: http://localhost:5000
# Run experiments
python src/train.py
```

**Tracked per experiment:**
- Parameters: `model_type`, `n_estimators`, `max_depth`, `learning_rate`
- Metrics: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`
- Artifacts: trained model, confusion matrix, feature importance plot

---

## 🤗 Hugging Face Spaces Deployment

```bash
# 1. Create a new Space at https://huggingface.co/new-space
#    SDK: Gradio, Space name: flight-delay-predictor

# 2. Clone the Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/flight-delay-predictor
cd flight-delay-predictor

# 3. Copy project files
cp app.py requirements.txt README.md ./
cp -r src/ models/ ./

# 4. Push to deploy
git add .
git commit -m "Deploy flight delay predictor"
git push

# Live in ~2 minutes at:
# https://huggingface.co/spaces/YOUR_USERNAME/flight-delay-predictor
```

---

## 📤 GitHub Upload

```bash
git init
git add .
git commit -m "Initial commit: Flight Delay Prediction ML System"
git remote add origin https://github.com/YOUR_USERNAME/flight-delay-prediction.git
git branch -M main
git push -u origin main
```

---

## 🔑 Features Used for Prediction

| Feature | Description |
|---------|-------------|
| `MONTH` | Month of flight |
| `DAY_OF_WEEK` | Day of week (1=Mon) |
| `DEP_HOUR` | Scheduled departure hour |
| `IS_WEEKEND` | Weekend flag (0/1) |
| `IS_RUSH_HOUR` | Rush hour flag (0/1) |
| `SEASON` | Season (1=Winter...4=Fall) |
| `OP_CARRIER_CODE` | Airline code (encoded) |
| `ORIGIN_CODE` | Origin airport (encoded) |
| `DEST_CODE` | Destination airport (encoded)|
| `DISTANCE` | Flight distance in miles |
| `CRS_ELAPSED_TIME` | Scheduled flight duration |

**Target:** `DELAYED = 1` if `ARR_DELAY > 15 minutes`, else `0`

---

## 📈 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~0.72 | ~0.68 | ~0.76 |
| Random Forest | ~0.78 | ~0.74 | ~0.83 |
| XGBoost | ~0.81 | ~0.77 | ~0.87 |
| **LightGBM** | **~0.82** | **~0.79** | **~0.88** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data** | Pandas, NumPy, SQLAlchemy (SQLite) |
| **ML** | Scikit-learn, XGBoost, LightGBM |
| **MLOps** | MLflow, Evidently AI |
| **API** | FastAPI, Uvicorn, Pydantic |
| **UI** | Gradio |
| **Container** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Cloud** | Hugging Face Spaces |

---

## ❓ Key Project Q&A

**Q: What is the delay threshold used?**
A: A flight is classified as *delayed* if `ARR_DELAY > 15 minutes` (standard industry definition by FAA/BTS).

**Q: How is class imbalance handled?**
A: Via `class_weight='balanced'` in LightGBM and `scale_pos_weight` in XGBoost.

**Q: Why not predict exact delay minutes?**
A: Binary classification is interpretable, actionable, and matches real-world usage (passengers want "will I be late?", not exact minutes).

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Your Name** | MSc Data Science / ML Engineering
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Hugging Face: [@YOUR_USERNAME](https://huggingface.co/YOUR_USERNAME)

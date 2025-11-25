# PlayStudios Fraud Detection API

FastAPI-based REST API for real-time credit card fraud detection using TensorFlow deep learning model trained on Kaggle Credit Card Fraud Detection dataset.

## 📋 Overview

Production-ready fraud detection API serving predictions from a pre-trained neural network. The model analyzes 30 transaction features and returns binary fraud classification (0 = legitimate, 1 = fraud).

## 🏗 Architecture

```
src/
├── api/                          # API Layer
│   ├── main.py                   # FastAPI app, CORS, router setup
│   ├── models.py                 # Pydantic request/response models
│   └── routes/
│       └── fraud_detection.py    # POST /detect endpoint
├── core/                         # Business Logic Layer
│   └── fraud_model.py            # Model loading, HuggingFace integration, inference
└── utils/                        # Utilities Layer
    ├── config.py                 # Environment config (Pydantic Settings)
    └── logging.py                # Logging setup
```

### Key Components

**1. API Layer** (`src/api/`)
- `main.py`: FastAPI application initialization, CORS middleware, router registration
- `models.py`: Type-safe request/response schemas with Pydantic
- `routes/fraud_detection.py`: Fraud detection endpoint with error handling

**2. Core Layer** (`src/core/`)
- `fraud_model.py`: 
  - Automatic model download from HuggingFace Hub
  - TensorFlow model initialization with TFSMLayer
  - Inference logic with configurable threshold
  - Fallback mechanism for model loading

**3. Utils Layer** (`src/utils/`)
- `config.py`: Centralized configuration using Pydantic Settings and dotenv
- `logging.py`: Structured logging configuration

### Data Flow

```
Client Request
    ↓
POST /detect (fraud_detection.py)
    ↓
FraudDetectionRequest (Pydantic validation)
    ↓
FraudDetectionModel.predict() (fraud_model.py)
    ↓
TensorFlow Model Inference
    ↓
Threshold Application (FRAUD_THRESHOLD)
    ↓
FraudDetectionResponse
    ↓
JSON Response to Client
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- uv (recommended) or pip

### Installation & Setup

1. **Clone and navigate to project**:
```bash
git clone <repository-url>
cd playSTUDIOS-test
```

2. **Create `.env` file**:
```env
FRAUD_MODEL_LOCAL_PATH=./hf_imbalanced_model
FRAUD_MODEL_REPO_ID=your-huggingface-repo/model
FRAUD_THRESHOLD=0.5
HOST=0.0.0.0
PORT=8000
```

3. **Run server** (uv automatically installs dependencies):
```bash
uv run python -m src.api.main
```

Server will be available at `http://localhost:8000`

**Interactive Docs**: http://localhost:8000/docs

## 📡 API Usage

### Endpoint: POST `/detect`

**Request**:
```json
{
  "features": [0.0, -1.36, -0.07, 2.54, ..., 149.62]  // 30 float values
}
```

**Features**: Time, V1-V28 (PCA components), Amount

**Response**:
```json
{
  "result": [0]  // 0 = legitimate, 1 = fraud
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, -1.36, ..., 149.62]}'
```

## 🧪 Testing

### Run Test Suite

```bash
# Start server
uv run python -m src.api.main

# In another terminal, run tests
uv run python tests/test_api.py --host localhost --port 8000
```

**Test Options**:
```bash
# Verbose output
uv run python tests/test_api.py -v

# Save results to JSON
uv run python tests/test_api.py --output results.json

# Custom test data
uv run python tests/test_api.py --csv path/to/test.csv
```

**Test Output**:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (TP, TN, FP, FN)
- Per-request validation against 100 Kaggle samples

## 📁 Project Structure

```
playSTUDIOS-test/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app entry point
│   │   ├── models.py               # Pydantic schemas
│   │   └── routes/
│   │       ├── __init__.py         # Router exports
│   │       └── fraud_detection.py  # /detect endpoint handler
│   ├── core/
│   │   └── fraud_model.py          # Model management & inference
│   └── utils/
│       ├── __init__.py             # Utility exports
│       ├── config.py               # Environment config
│       └── logging.py              # Logging setup
├── tests/
│   ├── test.csv                    # 100 Kaggle test samples
│   └── test_api.py                 # Automated test suite
├── hf_imbalanced_model/            # Cached TensorFlow model
├── .env                            # Environment variables
├── pyproject.toml                  # Dependencies & metadata
├── uv.lock                         # Dependency lock file
└── README.md
```

## ⚙️ Configuration

Environment variables in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `FRAUD_MODEL_LOCAL_PATH` | Model cache directory | `./hf_imbalanced_model` |
| `FRAUD_MODEL_REPO_ID` | HuggingFace repo | `username/model-name` |
| `FRAUD_THRESHOLD` | Classification threshold | `0.5` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

## 🤖 Model Details

- **Dataset**: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions (492 frauds, 0.172%)
- **Features**: 30 (Time, V1-V28 PCA components, Amount)
- **Model**: Deep Neural Network (TensorFlow/Keras)
- **Format**: SavedModel with TFSMLayer
- **Performance**: <50ms inference (warm), ~100-200 req/s

## 🔮 Future Work

### High Priority

1. **Apache Kafka Integration**
   - Kafka consumers for high-throughput transaction streams
   - Request rate limiting and load balancing
   - Batch inference through Kafka topics
   - Dead letter queues for failed predictions

2. **Database Logging**
   - PostgreSQL/MySQL for request/prediction logs
   - Audit trails (timestamp, features, prediction, confidence)
   - Data retention policies and indexed queries
   - Compliance and debugging support

3. **Monitoring**
   - Prometheus metrics (latency, throughput, errors)
   - Grafana dashboards
   - OpenTelemetry distributed tracing
   - Model drift detection and alerting

### Medium Priority

4. **API Enhancements**: Batch predictions, authentication, caching (Redis)
5. **Infrastructure**: Docker, Kubernetes, CI/CD pipeline
6. **Model Improvements**: A/B testing, explainability (SHAP/LIME), ensemble models

### Low Priority

7. **Security**: Request signing, rate limiting per API key, input sanitization
8. **Scalability**: Horizontal scaling, model optimization (TensorRT/ONNX)

## 🐛 Troubleshooting

**Port in use**:
```bash
lsof -ti:8000 | xargs kill -9
```

**Model loading fails**: Check `.env` config and internet connectivity for HuggingFace download

**Import errors**: Ensure `uv` is installed or run `pip install -e .`

---

**Note**: This is a demonstration project for PlayStudios Backend MLE Home Assignment showcasing production-ready practices for ML API development.

from fastapi import APIRouter, HTTPException
import numpy as np

from src.api.models import FraudDetectionRequest, FraudDetectionResponse
from src.core.fraud_model import FraudDetectionModel
from src.utils import get_logger

logger = get_logger(__name__)

fraud_model = FraudDetectionModel()

router = APIRouter(tags=["fraud_detection"])

@router.post("/detect")
async def detect_fraud(request: FraudDetectionRequest):
    try:
        logger.info(f"Detecting fraud for features: {len(request.features)} features")
        input_data = np.array(request.features).astype("float32")
        result = fraud_model.predict(input_data)
        logger.info(f"Fraud detection result: {result}")
        return FraudDetectionResponse(result=result)
    except Exception as e:
        logger.error(f"Error detecting fraud: {e}")
        raise HTTPException(status_code=500, detail=str(e))

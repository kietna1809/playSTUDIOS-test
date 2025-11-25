from pydantic import BaseModel

class FraudDetectionRequest(BaseModel):
    features: list[list[float]]      

class FraudDetectionResponse(BaseModel):
    result: list[int]
    
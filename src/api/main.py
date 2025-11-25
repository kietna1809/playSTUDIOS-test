from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.routes import fraud_detection_router
from src.utils import ai_config

app = FastAPI(
    title="PlayStudios Fraud Detection API",
    description="API for fraud detection",
    version="0.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(fraud_detection_router)

if __name__ == "__main__":
    uvicorn.run(app, host=ai_config.HOST, port=ai_config.PORT)
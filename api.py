from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import predict_report

app = FastAPI(
    title="Clinical Report Classification API",
    version="1.0"
)


class ReportRequest(BaseModel):
    report: str = Field(
        min_length=1,
        max_length=20000
    )


class PredictionItem(BaseModel):
    specialty: str
    confidence: float


class PredictionResponse(BaseModel):
    status: str
    message: Optional[str] = None
    specialty: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: List[PredictionItem] = Field(default_factory=list)
    readability_score: Optional[float] = None
    complexity: Optional[str] = None
    medical_score: Optional[int] = None


@app.get("/")
def home():
    return {
        "message": "Clinical Report Classification API Running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ReportRequest):
    return predict_report(data.report)
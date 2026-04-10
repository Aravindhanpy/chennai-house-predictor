import pickle
import json
import numpy as np
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# Load model and columns on startup
with open("chennai_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("columns.json") as f:
    COLUMNS = json.load(f)["data_columns"]

app = FastAPI(title="Chennai House Price Predictor")

# Allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request body schema
class PredictRequest(BaseModel):
    location: str
    builder: str = ""
    bhk: int
    bathroom: int
    area: float
    age: int


# Request body schema
class PredictResponse(BaseModel):
    price: float
    price_low: float
    price_high: float
    price_formatted: str
    range_formatted: str


def fmt(n: float) -> str:
    if n >= 100:
        return f"₹{n/100:.2f} Cr"
    return f"₹{n:.2f}L"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Build feature vector (all zeros)
    x = np.zeros(len(COLUMNS))

    x[COLUMNS.index("area")]     = req.area
    x[COLUMNS.index("bhk")]      = req.bhk
    x[COLUMNS.index("bathroom")] = req.bathroom
    x[COLUMNS.index("age")]      = req.age

    # One-hot: location
    loc = req.location.strip().lower()
    if loc in COLUMNS:
        x[COLUMNS.index(loc)] = 1

    # One-hot: builder
    bld = req.builder.strip().lower()
    if bld and bld in COLUMNS:
        x[COLUMNS.index(bld)] = 1

    # Predict
    price = float(model.predict([x])[0])
    price = max(5.0, price)

    price_low  = round(price * 0.90, 2)
    price_high = round(price * 1.10, 2)
    price      = round(price, 2)

    return PredictResponse(
        price=price,
        price_low=price_low,
        price_high=price_high,
        price_formatted=fmt(price),
        range_formatted=f"{fmt(price_low)} – {fmt(price_high)}",
    )


@app.get("/health")
def health():
    return {"status": "ok", "columns": len(COLUMNS)}


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

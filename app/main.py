from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Customer Churn Predictor")

model = joblib.load("model.pkl")

FEATURE_COLUMNS = [
    "gender",
    "age",
    "discount_ratio",
    "avg_quantity_per_product",
    "inactivity_ratio",
    "spend_per_product",
    "purchases_per_day",
    "large_cart",
    "high_spender",
]


class UserFeatures(BaseModel):
    gender: int                      # 0 = female, 1 = male
    age: int
    discount_ratio: float            # (total - discounted) / total
    avg_quantity_per_product: float  # total_quantity / total_products
    inactivity_ratio: float          # days_inactive / days_since_registration
    spend_per_product: float         # total_amount / total_products
    purchases_per_day: float         # total_quantity / days_since_registration
    large_cart: int                  # 0 or 1
    high_spender: int                # 0 or 1


@app.get("/")
def root():
    return {"message": "Customer Churn Predictor API — visit /docs to test"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"features": FEATURE_COLUMNS}


@app.post("/predict")
def predict(user: UserFeatures):

    input_data = np.array([[
        user.gender,
        user.age,
        user.discount_ratio,
        user.avg_quantity_per_product,
        user.inactivity_ratio,
        user.spend_per_product,
        user.purchases_per_day,
        user.large_cart,
        user.high_spender,
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]


    if probability >= 0.75:
        recommendation = "High churn risk. Trigger immediate re-engagement campaign with a time-limited discount offer."
    elif probability >= 0.5:
        recommendation = "Moderate churn risk. Send a personalized product recommendation email to re-activate interest."
    elif probability >= 0.25:
        recommendation = "Low churn risk. Enroll user in loyalty program to reinforce engagement."
    else:
        recommendation = "User is active and healthy. No intervention needed at this time."

    return {
        "churned": bool(prediction),
        "churn_probability": round(float(probability), 3),
        "recommendation": recommendation 
    }



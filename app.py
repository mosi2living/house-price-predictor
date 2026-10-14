import streamlit as st
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(
    page_title="🏠 House Price Predictor",
    layout="centered"
)
# Page title
# st.title("🏠 House Price Predictor - TC8")

# Header
st.title("🏠 House Price Predictor - TC8")
st.caption(
    "An XGBoost model trained on the California Housing dataset "
    
)


# Load data
@st.cache_resource
def load_everything():
    model = XGBRegressor()
    model.load_model("best_model.json")
    scaler = joblib.load("scaler.pkl")
    winsor_cap = joblib.load("winsor_cap.pkl")
    meta = joblib.load("meta.pkl")
    return model, scaler, winsor_cap, meta

model, scaler, winsor_cap, meta = load_everything()
FEATURES = meta["feature_names"]

#model, scaler, winsor_cap = load_everything()

#sidebar
with st.sidebar:
    st.header("About this model")
    st.metric("Test R² Score", f"{meta['r2_score']:.3f}")
    st.metric("Test MAE", f"{meta['mae']:.3f}")
    st.metric("Test RMSE", f"{meta['rmse']:.3f}")

    st.write(
        """
        - **Inputs:** 8 features of a California block group  
        - **Output:** Estimated median house value  
        - **Model:** Tuned XGBoost Regressor  
        - **Preprocessing:** Winsorization + Standard Scaling
        """
    )
    st.info(
        "Capstone Project for TechCrush C8 AI/ML Class. "
        "Predictions are illustrative only."
    )


# Input section
st.subheader("House Features")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income (MedInc)", min_value=0.0, max_value=15.0, value=4.0, step=0.1, help="Income (in tens of thousands of USD)",)
    HouseAge = st.number_input("House Age (HouseAge)", min_value=1.0, max_value=52.0, value=25.0, step=1.0, help="House age",)
    AveRooms = st.number_input("Average Rooms (AveRooms)", min_value=1.0, max_value=15.0, value=5.0, step=0.1, help="Number of rooms per house",)
    AveBedrms = st.number_input("Average Bedrooms (AveBedrms)", min_value=0.5, max_value=5.0, value=1.0, step=0.1,help="Number of bedrooms per house",)

with col2:
    Population = st.number_input("Population", min_value=1.0, max_value=10000.0, value=1000.0, step=10.0, help="population",)
    AveOccup = st.number_input("Average Occupancy (AveOccup)", min_value=0.5, max_value=10.0, value=3.0, step=0.1, help="Number of Occupants",)
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.5, step=0.01, help="Latitude of the house",)
    Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.0, step=0.01, help="Longitude of the house",)


def predict(features: dict) -> float:

    
    new_house = pd.DataFrame([features], columns=[
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ])

    for col, limits in winsor_cap.items():
        new_house[col] = new_house[col].clip(limits["lower"], limits["upper"])

    scaled = scaler.transform(new_house)

    prediction = model.predict(scaled)[0]
    return prediction

# Prediction
if st.button("Predict House Price", type="primary", use_container_width=True):
    features = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    prediction = predict(features)
    predicted_price = prediction * 100000

    st.divider()
    st.success("Prediction Complete!")
    st.metric(label="Estimated House Value", value=f"${predicted_price:,.2f}")
    


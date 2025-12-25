import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load model
# -------------------------------
price = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered"
)

st.title("🚗 Car Price Predictor")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Car_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

car = load_data()


companies = sorted(car["company"].unique())
years = sorted(car["year"].unique(), reverse=True)
fuel_types = sorted(car["fuel_type"].unique())

company = st.selectbox("Select Company", companies)

models = sorted(car[car["company"] == company]["name"].unique())
car_name = st.selectbox("Select Model", models)

year = st.selectbox("Select Manufacturing Year", years)
fuel = st.selectbox("Select Fuel Type", fuel_types)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Price"):

    input_df = pd.DataFrame(
        [[car_name, company, year, kms_driven, fuel]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    predicted_price = price.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: ₹ {predicted_price:,.0f}")

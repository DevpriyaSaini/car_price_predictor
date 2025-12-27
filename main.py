import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page config (MUST be first)
# -------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("LinearRegressionModel.pkl", "rb"))

price = load_model()

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Car_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

car = load_data()

# -------------------------------
# Header
# -------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🚗 Car Price Predictor</h1>
    <p style="text-align:center; color:gray;">
    Estimate the resale price of your car using machine learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🔧 Car Details")

companies = sorted(car["company"].unique())
years = sorted(car["year"].unique(), reverse=True)
fuel_types = sorted(car["fuel_type"].unique())

company = st.sidebar.selectbox("Company", companies)

models = sorted(car[car["company"] == company]["name"].unique())
car_name = st.sidebar.selectbox("Model", models)

year = st.sidebar.selectbox("Manufacturing Year", years)

fuel = st.sidebar.selectbox("Fuel Type", fuel_types)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500_000,
    step=1_000,
    help="Total distance driven in kilometers"
)

# -------------------------------
# Main Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("🚘 Selected Car", car_name)

with col2:
    st.metric("🏭 Company", company)

st.divider()

# -------------------------------
# Predict Button
# -------------------------------
if st.button("💰 Predict Price", use_container_width=True):

    input_df = pd.DataFrame(
        [[car_name, company, year, kms_driven, fuel]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    predicted_price = price.predict(input_df)[0]

    st.success(
        f"""
        ### 💸 Estimated Price  
        **₹ {predicted_price:,.0f}**
        """
    )

    st.caption("⚠️ This is an estimated value based on historical data.")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:14px;">
    Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

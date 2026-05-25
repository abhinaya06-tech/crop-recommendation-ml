import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/crop_data.csv")
model = joblib.load("model/model.pkl")

# ---------------- HEADER ----------------
st.title("🌾 Smart Crop Recommendation System")

st.markdown("""
Predict the most suitable crops based on soil nutrients and environmental conditions using Machine Learning.
""")

# ---------------- METRICS ----------------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Dataset Size", f"{df.shape[0]} Rows")

with metric2:
    st.metric("Features", f"{df.shape[1]-1}")

with metric3:
    st.metric("Crop Types", f"{df['label'].nunique()}")

st.divider()

# ---------------- INSIGHTS ----------------
st.subheader("📌 Key Insights")

st.info("""
• Rainfall and humidity are dominant factors  
• Soil nutrients (N, P, K) strongly influence crop selection  
• Random Forest provides probabilistic top-3 predictions  
""")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "🌱 Prediction",
    "📊 Visualizations",
    "ℹ About Model"
])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:

    st.subheader("🧪 Enter Soil & Climate Conditions")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=60.0
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=35.0
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=35.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            value=60.0
        )

        ph = st.number_input(
            "pH Value",
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            value=200.0
        )

    st.write("")

    # ---------------- PREDICTION ----------------
    if st.button("🌱 Predict Suitable Crops"):

        data = [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]]

        probs = model.predict_proba(data)[0]
        classes = model.classes_

        top3 = np.argsort(probs)[-3:][::-1]

        st.success("Top 3 Recommended Crops")

        for idx, i in enumerate(top3):
            st.success(
                f"#{idx+1} Recommended Crop: "
                f"{classes[i].capitalize()} "
                f"({probs[i]*100:.2f}%)"
            )

    st.divider()

    # ---------------- TELANGANA SECTION ----------------
    st.subheader("🌦 Telangana Simulation")

    st.write(
        "Use estimated Telangana environmental averages for prediction."
    )

    telangana_data = {
        "N": 60,
        "P": 35,
        "K": 35,
        "temperature": 33,
        "humidity": 55,
        "ph": 7.2,
        "rainfall": 600
    }

    if st.button("Predict for Telangana"):

        data = [[
            telangana_data["N"],
            telangana_data["P"],
            telangana_data["K"],
            telangana_data["temperature"],
            telangana_data["humidity"],
            telangana_data["ph"],
            telangana_data["rainfall"]
        ]]

        probs = model.predict_proba(data)[0]
        classes = model.classes_

        top3 = np.argsort(probs)[-3:][::-1]

        st.success("Top 3 Crops for Telangana")

        for idx, i in enumerate(top3):
            st.success(
                f"#{idx+1} Recommended Crop: "
                f"{classes[i].capitalize()} "
                f"({probs[i]*100:.2f}%)"
            )

        st.warning("""
Predictions are based on generalized dataset averages and may vary from real agricultural conditions.
""")

# =========================================================
# TAB 2 - VISUALIZATIONS
# =========================================================
with tab2:

    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("📊 Crop Distribution")

        fig, ax = plt.subplots(figsize=(12, 5))

        df["label"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

    with chart2:
        st.subheader("📈 Feature Importance")

        importances = model.feature_importances_

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.bar(
            df.drop("label", axis=1).columns,
            importances
        )

        plt.xticks(rotation=20)
        plt.tight_layout()

        st.pyplot(fig2)

# =========================================================
# TAB 3 - ABOUT MODEL
# =========================================================
with tab3:

    st.subheader("🤖 Model Information")

    st.write("""
### Algorithm Used
Random Forest Classifier

### Why Random Forest?
- Handles non-linear relationships effectively
- Works well with agricultural datasets
- Reduces overfitting compared to Decision Trees
- Provides feature importance analysis

### Input Features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

### Output
Top 3 most suitable crops with prediction probabilities.
""")
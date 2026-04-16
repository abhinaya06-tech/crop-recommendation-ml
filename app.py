import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv("data/crop_data.csv")
model = joblib.load("model/model.pkl")

# ------------------- TITLE -------------------
st.title("Crop Recommendation System")

# ------------------- INSIGHTS -------------------
st.markdown("### Key Insights from Data")
st.write("- Rainfall and humidity are dominant factors")
st.write("- Soil nutrients (N, P, K) also influence crop selection")
st.write("- Model provides probabilistic recommendations (Top 3 crops)")

# ------------------- DATA VISUALIZATION -------------------
st.subheader("Crop Distribution")
fig, ax = plt.subplots()
df["label"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

# ------------------- FEATURE IMPORTANCE -------------------
st.subheader("Feature Importance")
importances = model.feature_importances_

fig2, ax2 = plt.subplots()
ax2.bar(df.drop("label", axis=1).columns, importances)
plt.xticks(rotation=45)
st.pyplot(fig2)

# ------------------- TELANGANA DEFAULT -------------------
telangana_data = {
    "N": 60,
    "P": 35,
    "K": 35,
    "temperature": 33,
    "humidity": 55,
    "ph": 7.2,
    "rainfall": 600
}

st.info("Enter soil nutrients (N, P, K) and environmental conditions to get crop recommendations.")

# ------------------- MANUAL INPUT -------------------
st.subheader("Manual Input")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict"):
    if N == 0 and P == 0 and K == 0:
        st.error("Please enter valid input values.")
    else:
        data = [[N, P, K, temperature, humidity, ph, rainfall]]
        
        probs = model.predict_proba(data)[0]
        classes = model.classes_
        
        top3 = np.argsort(probs)[-3:][::-1]
        
        st.success("Top 3 Suitable Crops:")
        for i in top3:
            st.write(f"{classes[i]} ({probs[i]*100:.2f}%)")

# ------------------- TELANGANA MODE -------------------
st.subheader("Use Telangana Average Data")

if st.button("Predict for Telangana (2026 Simulation)"):
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
    
    st.success("Top 3 Suitable Crops for Telangana:")
    for i in top3:
        st.write(f"{classes[i]} ({probs[i]*100:.2f}%)")

    st.warning("Note: This recommendation is based on a generalized dataset and may not reflect region-specific agricultural practices.")
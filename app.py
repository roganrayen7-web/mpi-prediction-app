import streamlit as st

st.set_page_config(layout="wide")
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("MPI Prediction Dashboard")

st.sidebar.header("Input Features")

income = st.sidebar.slider("Income Index", 0.0, 1.0, 0.5)
ger = st.sidebar.slider("Female Secondary GER", 0.0, 100.0, 50.0)
sanitation = st.sidebar.slider("Sanitation %", 0.0, 100.0, 50.0)
infra = st.sidebar.slider("Infrastructure Spend (log)", 0.0, 15.0, 5.0)
rural = st.sidebar.slider("Rural Population % (log)", 0.0, 5.0, 2.0)

input_data = np.array([[income, ger, sanitation, infra, rural]])
scaled_input = scaler.transform(input_data)

lr_pred = lr_model.predict(scaled_input)[0]
rf_pred = rf_model.predict(scaled_input)[0]

st.subheader("Predicted MPI Value")
st.write(f"Linear Regression: {round(lr_pred, 3)}")
st.write(f"Random Forest: {round(rf_pred, 3)}")

features = ["income_index", "female_sec_ger", "sanitation_pct", "infra_spend", "rural_pop"]

st.subheader("Feature Importance (Linear Regression)")
fig, ax = plt.subplots()
ax.bar(features, lr_model.coef_)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Feature Importance (Random Forest)")
fig2, ax2 = plt.subplots()
ax2.bar(features, rf_model.feature_importances_)
plt.xticks(rotation=45)
st.pyplot(fig2)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Raisin Class Predictor",
    page_icon="🍇",
    layout="centered"
)

# ----------------------------------------------------------------------
# UI Styling (ONLY UI IMPROVEMENT)
# ----------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    h1 {
        color: #22c55e;
        text-align: center;
    }

    .stButton>button {
        background-color: #22c55e;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-size: 18px;
    }

    .stButton>button:hover {
        background-color: #16a34a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
st.markdown("<h1>🍇 Raisin AI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Machine Learning powered raisin classification system</p>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Data Loading & Model Training (UNCHANGED)
# ----------------------------------------------------------------------
@st.cache_data
def load_and_train_model():
    try:
        df = pd.read_csv("Raisin_Dataset.csv")
    except FileNotFoundError:
        st.error("Please ensure 'Raisin_Dataset.csv' is in the same directory.")
        st.stop()

    df["Class"] = df["Class"].map({"Kecimen": 1, "Besni": 0})

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.1
    )

    model = RandomForestClassifier(
        n_estimators=10,
        criterion="gini",
        max_depth=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    test_acc = model.score(X_test, y_test)

    return model, test_acc, X


model, test_accuracy, X_features = load_and_train_model()

# ----------------------------------------------------------------------
# Sidebar UI (IMPROVED ONLY)
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Feature Controls")
st.sidebar.markdown("Adjust raisin parameters 👇")

def user_input_features():
    st.sidebar.markdown("### 📏 Size Features")

    area = st.sidebar.number_input("Area", 0, 300000, 80000)
    perimeter = st.sidebar.number_input("Perimeter", 0.0, 3000.0, 1100.0)

    st.sidebar.markdown("### 📐 Shape Features")

    major_axis = st.sidebar.number_input("MajorAxisLength", 0.0, 1000.0, 400.0)
    minor_axis = st.sidebar.number_input("MinorAxisLength", 0.0, 1000.0, 250.0)
    eccentricity = st.sidebar.slider("Eccentricity", 0.0, 1.0, 0.75)

    st.sidebar.markdown("### 📦 Area Features")

    convex_area = st.sidebar.number_input("ConvexArea", 0, 300000, 85000)
    extent = st.sidebar.slider("Extent", 0.0, 1.0, 0.70)

    data = {
        "Area": area,
        "MajorAxisLength": major_axis,
        "MinorAxisLength": minor_axis,
        "Eccentricity": eccentricity,
        "ConvexArea": convex_area,
        "Extent": extent,
        "Perimeter": perimeter
    }

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# ----------------------------------------------------------------------
# Sidebar Metrics
# ----------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.metric("Model Accuracy", f"{test_accuracy:.2%}")

# ----------------------------------------------------------------------
# Main UI Output
# ----------------------------------------------------------------------
st.subheader("📊 Selected Input Features")
st.dataframe(input_df)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if st.button("🔮 Predict Class"):
    input_df = input_df[X_features.columns]

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("🍇 This raisin is classified as **Kecimen**")
    else:
        st.error("🍇 This raisin is classified as **Besni**")

    st.markdown("### 🔥 Confidence Scores")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Besni", f"{prediction_proba[0]:.2%}")

    with col2:
        st.metric("Kecimen", f"{prediction_proba[1]:.2%}")

    # Chart
    st.markdown("### 📊 Probability Chart")

    prob_df = pd.DataFrame({
        "Class": ["Besni", "Kecimen"],
        "Probability": [prediction_proba[0], prediction_proba[1]]
    })

    st.bar_chart(prob_df.set_index("Class"))
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Page config
st.set_page_config(page_title="Dry Bean Classification ML Dashboard")
st.title("Dry Bean Classification ML Dashboard")
st.markdown("Compare multiple Machine Learning models on the Dry Bean Dataset")

# Load resources (cached)
@st.cache_resource
def load_resources():
    models = {
        "Logistic Regression": joblib.load("model/Logistic Regression.pkl"),
        "Decision Tree": joblib.load("model/Decision Tree.pkl"),
        "KNN": joblib.load("model/KNN.pkl"),
        "Naive Bayes": joblib.load("model/Naive Bayes.pkl"),
        "Random Forest": joblib.load("model/Random Forest.pkl"),
        "XGBoost": joblib.load("model/XGBoost.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    metrics = pd.read_csv("model/metrics.csv")
    return models, scaler, label_encoder, metrics

models, scaler, le, metrics = load_resources()

# Sidebar
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)
uploaded = st.sidebar.file_uploader("Upload Test CSV", type="csv")
run_btn = st.sidebar.button("Run Evaluation")

# Evaluation
if uploaded and run_btn:
    df = pd.read_csv(uploaded)

    if "Class" not in df.columns:
        st.error("CSV must contain 'Class' column")
        st.stop()

    X = df.drop("Class", axis=1)
    y = le.transform(df["Class"])

    # scale features
    X = scaler.transform(X)
    model = models[model_name]
    preds = model.predict(X)

    # ---- Metrics ----
    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    # ---- Confusion Matrix ----
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

# Comparison table
st.subheader("Model Comparison Metrics")
st.dataframe(metrics)

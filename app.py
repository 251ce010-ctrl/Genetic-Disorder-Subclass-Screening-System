
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Genetic Disorder Screening",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Genetic Disorder Subclass Screening")
st.write("AI-Based Genetic Disorder Prediction System")
st.info("Enter the patient details below to get a model prediction.")
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "train_genetic_disorders.csv")


FEATURES = [
    "Patient Age",
    "Gender",
    "Genes in mother's side",
    "Inherited from father",
    "Maternal gene",
    "Paternal gene",
    "Blood test result",
    "Birth defects",
    "Symptom 1",
    "Symptom 2",
    "Symptom 3",
    "Symptom 4",
    "Symptom 5"
]

TARGET = "Disorder Subclass"


# ---------------- TRAIN MODEL ----------------

@st.cache_resource
def train_model():

    df = pd.read_csv(DATASET)

    # Check required columns
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]

    if missing:
        raise ValueError("Missing dataset columns: " + str(missing))

    # Keep only required columns
    df = df[FEATURES + [TARGET]].copy()

    # Remove rows without target
    df = df.dropna(subset=[TARGET])

    # Clean target text
    df[TARGET] = df[TARGET].astype(str).str.strip()

    # Remove empty target values
    df = df[df[TARGET] != ""]

    X = df[FEATURES]
    y = df[TARGET]

    numeric_features = ["Patient Age"]

    categorical_features = [
        c for c in FEATURES if c not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X, y)

    # Find the most common genetic disorder for each subclass
    disorder_map = (
        pd.read_csv(DATASET)
        .dropna(subset=["Disorder Subclass", "Genetic Disorder"])
        .groupby("Disorder Subclass")["Genetic Disorder"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    return model, disorder_map, len(df)


try:
    with st.spinner("Loading and training the AI model..."):
        model, disorder_map, total_rows = train_model()

    st.success("AI model is ready!")

except Exception as e:
    st.error("Unable to load the model or dataset.")
    st.write("Error:", str(e))
    st.stop()


# ---------------- PATIENT INPUT FORM ----------------

st.subheader("Patient Details")

with st.form("prediction_form"):

    age = st.number_input(
        "Patient Age",
        min_value=0,
        max_value=120,
        value=12
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    maternal_side = st.selectbox(
        "Genes in mother's side",
        ["Yes", "No"]
    )

    inherited_father = st.selectbox(
        "Inherited from father",
        ["Yes", "No"]
    )

    maternal_gene = st.selectbox(
        "Maternal gene",
        ["Yes", "No"]
    )

    paternal_gene = st.selectbox(
        "Paternal gene",
        ["Yes", "No"]
    )

    blood_test = st.selectbox(
        "Blood test result",
        ["normal", "abnormal", "inconclusive"]
    )

    birth_defects = st.selectbox(
        "Birth defects",
        ["Singular", "Multiple"]
    )

    st.subheader("Symptoms")

    symptom_values = []

    for i in range(1, 6):
        answer = st.selectbox(
            f"Symptom {i}",
            ["No", "Yes"],
            key=f"symptom_{i}"
        )

        symptom_values.append(1 if answer == "Yes" else 0)

    submitted = st.form_submit_button(
        "🔍 Predict Genetic Disorder",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submitted:

    patient_data = pd.DataFrame([{
        "Patient Age": age,
        "Gender": gender,
        "Genes in mother's side": maternal_side,
        "Inherited from father": inherited_father,
        "Maternal gene": maternal_gene,
        "Paternal gene": paternal_gene,
        "Blood test result": blood_test,
        "Birth defects": birth_defects,
        "Symptom 1": symptom_values[0],
        "Symptom 2": symptom_values[1],
        "Symptom 3": symptom_values[2],
        "Symptom 4": symptom_values[3],
        "Symptom 5": symptom_values[4]
    }])

    prediction = model.predict(patient_data)[0]

    genetic_disorder = disorder_map.get(
        prediction,
        "Not available"
    )

    st.divider()
    st.subheader("Prediction Result")

    st.success("Prediction completed!")

    st.write("**Predicted Genetic Disorder:**")
    st.write(genetic_disorder)

    st.write("**Predicted Disorder Subclass:**")
    st.write(prediction)

    st.caption(
        "This is an educational machine-learning demonstration, "
        "not a medical diagnosis. Consult a qualified healthcare professional."
    )

st.divider()
st.caption(f"Training records used: {total_rows:,}")
st.caption("Genetic Disorder Subclass Screening System")

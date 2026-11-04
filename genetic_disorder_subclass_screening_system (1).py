# -*- coding: utf-8 -*-
"""
Genetic Disorder Subclass Screening System
Streamlit application using Linear, Polynomial and RBF SVM.
"""

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

st.set_page_config(
    page_title="Genetic Disorder Subclass Screening System",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Genetic Disorder Subclass Screening System")
st.write(
    "This application trains SVM models to predict the genetic disorder subclass. "
    "Upload a labelled training CSV file to begin."
)

TARGET = "Disorder Subclass"

DROP_COLUMNS = [
    "Patient Id",
    "Patient First Name",
    "Family Name",
    "Father's name",
    "Institute Name",
    "Location of Institute",
    "Genetic Disorder",
]


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Clean invalid values, missing target rows and duplicate rows."""
    data = data.copy()
    data.columns = data.columns.str.strip()
    data = data.replace([-99, "-99"], np.nan)

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {', '.join(data.columns)}"
        )

    data = data.dropna(subset=[TARGET])
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)

    # Remove identifier/personal/inappropriate columns when present.
    data = data.drop(columns=DROP_COLUMNS, errors="ignore")

    if data.empty:
        raise ValueError("No rows remain after removing missing target values.")

    if data[TARGET].nunique() < 2:
        raise ValueError("Training data must contain at least two disorder subclasses.")

    return data


def build_pipeline(X: pd.DataFrame, kernel: str) -> Pipeline:
    """Build preprocessing + feature selection + SVM pipeline."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("The dataset has no usable feature columns.")

    preprocessor = ColumnTransformer(transformers=transformers)

    # Select no more than 20 features, and never select fewer than 1.
    # The actual transformed feature count is not known until preprocessing,
    # so SelectKBest uses k='all' when the requested k cannot be determined
    # from the raw columns. This avoids failures on small datasets.
    requested_k = min(20, max(1, X.shape[1]))

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("selection", SelectKBest(score_func=f_classif, k=requested_k)),
            ("classifier", SVC(kernel=kernel, cache_size=1000)),
        ]
    )


def safe_train_test_split(X, y):
    """Use stratification when every class has enough examples."""
    counts = y.value_counts()
    can_stratify = len(counts) > 1 and counts.min() >= 2

    if can_stratify:
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    st.warning(
        "Some classes have fewer than 2 rows. The data is split without stratification."
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


training_file = st.file_uploader(
    "Step 1 – Upload Training Dataset (CSV)",
    type=["csv"],
    key="training_csv",
)

if training_file is None:
    st.info("Please upload your training CSV file. The CSV must contain a 'Disorder Subclass' column.")
    st.stop()

try:
    raw_df = pd.read_csv(training_file)
    df = clean_dataset(raw_df)
except Exception as error:
    st.error(f"Could not load or clean the training dataset: {error}")
    st.stop()

st.subheader("Step 2 – Dataset Information")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]:,}")
col3.metric("Disorder subclasses", int(df[TARGET].nunique()))

with st.expander("View dataset preview"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View missing values and class counts"):
    st.write("Missing values:")
    st.dataframe(df.isnull().sum().rename("Missing values"))
    st.write("Disorder subclass counts:")
    st.dataframe(df[TARGET].value_counts(dropna=False))

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

if X.shape[1] == 0:
    st.error("No feature columns remain after removing unwanted columns.")
    st.stop()

# Remove columns that contain no information.
constant_columns = [
    column for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]
if constant_columns:
    X = X.drop(columns=constant_columns)
    st.warning("Removed columns with only one unique value: " + ", ".join(constant_columns))

if X.shape[1] == 0:
    st.error("No useful feature columns are available for training.")
    st.stop()

try:
    X_train, X_val, y_train, y_val = safe_train_test_split(X, y)
except Exception as error:
    st.error(f"Could not split the dataset: {error}")
    st.stop()

st.subheader("Step 3 – Train SVM Models")

if st.button("Train Linear, Polynomial and RBF SVM", type="primary"):
    trained_models = {}
    result_rows = []
    predictions_by_kernel = {}
    progress = st.progress(0)
    status = st.empty()

    kernels = ["linear", "poly", "rbf"]

    try:
        for index, kernel in enumerate(kernels):
            status.write(f"Training {kernel.upper()} SVM...")
            model = build_pipeline(X_train, kernel)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            macro_f1 = f1_score(
                y_val, predictions, average="macro", zero_division=0
            )

            trained_models[kernel] = model
            predictions_by_kernel[kernel] = predictions
            result_rows.append(
                {
                    "Kernel": kernel.upper(),
                    "Accuracy (%)": accuracy * 100,
                    "Macro F1 (%)": macro_f1 * 100,
                }
            )
            progress.progress((index + 1) / len(kernels))

        results_df = pd.DataFrame(result_rows)
        best_index = results_df["Macro F1 (%)"].idxmax()
        best_kernel = ["linear", "poly", "rbf"][best_index]
        best_model = trained_models[best_kernel]

        st.session_state["trained_models"] = trained_models
        st.session_state["predictions_by_kernel"] = predictions_by_kernel
        st.session_state["results_df"] = results_df
        st.session_state["best_kernel"] = best_kernel
        st.session_state["best_model"] = best_model
        st.session_state["X_columns"] = X.columns.tolist()
        st.session_state["validation_y"] = y_val

        status.success("All three SVM models trained successfully!")

    except Exception as error:
        status.empty()
        st.error(f"Model training failed: {error}")

if "results_df" in st.session_state:
    st.subheader("Step 4 – Model Performance Comparison")
    st.dataframe(
        st.session_state["results_df"].style.format(
            {"Accuracy (%)": "{:.2f}", "Macro F1 (%)": "{:.2f}"}
        ),
        use_container_width=True,
    )

    best_kernel = st.session_state["best_kernel"]
    st.success(f"Best SVM kernel based on Macro F1: {best_kernel.upper()}")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=st.session_state["results_df"],
        x="Kernel",
        y="Accuracy (%)",
        ax=ax,
    )
    ax.set_title("SVM Kernel Accuracy Comparison")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Step 5 – Classification Report and Confusion Matrix")
    selected_kernel = st.selectbox(
        "Choose SVM kernel",
        ["linear", "poly", "rbf"],
        format_func=lambda value: value.upper(),
    )
    selected_model = st.session_state["trained_models"][selected_kernel]
    selected_predictions = st.session_state["predictions_by_kernel"][selected_kernel]
    validation_y = st.session_state["validation_y"]

    report = classification_report(
        validation_y,
        selected_predictions,
        zero_division=0,
    )
    st.text(report)

    labels = sorted(pd.Series(validation_y).astype(str).unique().tolist())
    cm = confusion_matrix(
        validation_y,
        selected_predictions,
        labels=labels,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    ).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{selected_kernel.upper()} SVM Confusion Matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Step 6 – Download Best Trained Model")
    model_bytes = io.BytesIO()
    joblib.dump(st.session_state["best_model"], model_bytes)
    model_bytes.seek(0)
    st.download_button(
        "Download Best SVM Model",
        data=model_bytes,
        file_name="genetic_disorder_model.pkl",
        mime="application/octet-stream",
    )

    st.subheader("Step 7 – Upload Test Dataset and Predict")
    test_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        key="test_csv",
    )

    if test_file is not None:
        try:
            test_df = pd.read_csv(test_file)
            test_df.columns = test_df.columns.str.strip()
            test_df = test_df.replace([-99, "-99"], np.nan)
            test_features = test_df.drop(
                columns=DROP_COLUMNS + [TARGET],
                errors="ignore",
            )

            # Ensure the test columns match the training columns and order.
            training_columns = st.session_state["X_columns"]
            test_features = test_features.reindex(columns=training_columns)

            test_predictions = st.session_state["best_model"].predict(test_features)

            prediction_df = pd.DataFrame(
                {"Predicted Disorder Subclass": test_predictions}
            )

            st.success("Prediction completed successfully!")
            st.dataframe(prediction_df.head(20), use_container_width=True)

            csv_bytes = prediction_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Results CSV",
                data=csv_bytes,
                file_name="genetic_disorder_predictions.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
else:
    st.info("Click 'Train Linear, Polynomial and RBF SVM' to train and compare the models.")


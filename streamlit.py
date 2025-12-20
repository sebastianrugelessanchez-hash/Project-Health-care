"""
Streamlit UI for the Healthcare forecasting project.
Upload the latest month data (features 2024-1 ... 2025-8) and get the next-month forecast (2025-9)
using the trained SVM artifact saved at tem_modelos/svm_model.joblib.
"""

from pathlib import Path
from typing import Tuple
import io
import os
import sys

import joblib
import pandas as pd
import streamlit as st


# ----------------------------
# Resolve import paths (Streamlit Cloud / Linux)
# ----------------------------
# Ensure local modules (e.g., processing.py) can be imported during joblib unpickling.
BASE_DIR = Path(__file__).resolve().parent  # repo root if streamlit.py is at root
PIPELINE_DIR = BASE_DIR / "Coding" / "pipeline"

if PIPELINE_DIR.exists():
    sys.path.insert(0, str(PIPELINE_DIR))
else:
    # Not fatal, but will likely break model loading if artifact depends on these modules
    # Keep the message visible to help debugging deployments
    pass


# Default artifact path: project root / tem_modelos / svm_model.joblib
DEFAULT_ARTIFACT_PATH = BASE_DIR / "tem_modelos" / "svm_model.joblib"
PRED_COLUMN = "prediction_2025-9"


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_artifact(artifact_path: Path):
    """Load the serialized model artifact."""
    return joblib.load(artifact_path)


def process_features(df: pd.DataFrame, artifact) -> pd.DataFrame:
    """
    Apply the same processing used in training.
    The processor inside the artifact is already fitted; keep fit=False.
    """
    return artifact["processor"].process_pipeline(
        df,
        artifact["config"],
        fit=False,
        select_usable=True,
        usable_columns=artifact["usable_columns"],
        remove_categorical=True,
    )


def predict(df: pd.DataFrame, artifact) -> pd.DataFrame:
    """Run predictions on a DataFrame using the loaded artifact."""
    processed = process_features(df, artifact)
    preds = artifact["model"].predict(processed)
    result = df.copy()
    result[PRED_COLUMN] = preds
    return result


def validate_columns(df: pd.DataFrame, usable_columns) -> Tuple[bool, str]:
    """Ensure the uploaded data has the required feature columns."""
    missing = [col for col in usable_columns if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, ""


def build_template(usable_columns) -> bytes:
    """Build a blank CSV template with the expected columns."""
    template = pd.DataFrame(columns=usable_columns)
    return template.to_csv(index=False).encode("utf-8")


def manual_input_form(usable_columns):
    """Render manual input form and return a single-row DataFrame when submitted."""
    with st.form("manual_input"):
        st.subheader("Manual entry")
        st.caption("Enter the last known values for 2024-1 ... 2025-8 to predict 2025-9.")
        values = {}
        for col in usable_columns:
            values[col] = st.number_input(col, value=0.0, step=1.0, format="%.4f")
        submitted = st.form_submit_button("Predict next month")
    if not submitted:
        return None
    return pd.DataFrame([values])


def file_upload_section():
    """Handle batch prediction via file upload."""
    st.subheader("Upload last-month data (CSV or Excel)")
    st.caption(
        "Upload a file with the required feature columns (2024-1 ... 2025-8). "
        "The app will return the forecast for 2025-9."
    )
    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if not uploaded:
        return None

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return None

    st.success(f"Loaded file with shape {df.shape}")
    st.dataframe(df.head(10))
    return df


def render_sidebar(artifact):
    usable_columns = artifact.get("usable_columns") or []
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Model: {artifact.get('model_name', 'SVM')}")
    st.sidebar.write(f"Features expected: {len(usable_columns)}")
    st.sidebar.write(", ".join(usable_columns))

    st.sidebar.markdown("---")
    st.sidebar.write("Template")
    csv_bytes = build_template(usable_columns)
    st.sidebar.download_button(
        "Download CSV template",
        data=csv_bytes,
        file_name="template_features.csv",
        mime="text/csv",
    )


# ----------------------------
# Page
# ----------------------------
def main():
    st.set_page_config(page_title="Healthcare Forecasting (SVM)", layout="wide")
    st.title("Predict Next-Month Demand (2025-9)")
    st.write(
        "Provide the latest month feature values (2024-1 through 2025-8). "
        "The trained SVM model will forecast the next month (2025-9)."
    )

    # Optional: keep this input for debugging, but default to the correct repo-relative path.
    artifact_path_str = st.text_input("Model artifact path", value=str(DEFAULT_ARTIFACT_PATH))
    artifact_path = Path(artifact_path_str).expanduser()

    # Helpful debug info for deployment
    with st.expander("Deployment debug info", expanded=False):
        st.write(f"BASE_DIR: {BASE_DIR}")
        st.write(f"PIPELINE_DIR: {PIPELINE_DIR} (exists={PIPELINE_DIR.exists()})")
        st.write(f"Artifact path: {artifact_path} (exists={artifact_path.exists()})")

    if not artifact_path.exists():
        st.error(
            f"Artifact not found at: {artifact_path}. "
            f"Expected something like: {DEFAULT_ARTIFACT_PATH}"
        )
        return

    try:
        artifact = load_artifact(artifact_path)
    except Exception as exc:
        st.error(f"Failed to load artifact: {exc}")
        st.info(
            "If the error mentions missing local modules (e.g., 'processing'), ensure the path "
            "Coding/pipeline is present and contains the corresponding .py files."
        )
        return

    usable_columns = artifact.get("usable_columns") or []
    if not usable_columns:
        st.error("Artifact is missing usable_columns; cannot build input form.")
        return

    render_sidebar(artifact)

    st.markdown("### Option 1: Upload a file")
    batch_df = file_upload_section()
    if batch_df is not None:
        ok, msg = validate_columns(batch_df, usable_columns)
        if not ok:
            st.error(msg)
        else:
            try:
                result = predict(batch_df, artifact)
                st.subheader("Predictions for 2025-9")
                st.dataframe(result)
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

    st.markdown("---")
    st.markdown("### Option 2: Enter values manually")
    single_df = manual_input_form(usable_columns)
    if single_df is not None:
        try:
            result = predict(single_df, artifact)
            st.metric("Predicted 2025-9", f"{result.iloc[0][PRED_COLUMN]:.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()


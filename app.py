""")
st.stop()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import io
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# joblib already imported above
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
ConfusionMatrixDisplay,
accuracy_score,
confusion_matrix,
f1_score,
precision_score,
recall_score,
roc_auc_score,
roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("diabetes_app")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FILE_NAME = "diabetes.csv"
MODEL_FILE_NAME = "best_model.joblib"
SCALER_FILE_NAME = "scaler.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
"Pregnancies",
"Glucose",
"BloodPressure",
"SkinThickness",
"Insulin",
"BMI",
"DiabetesPedigreeFunction",
"Age",
]
TARGET_COLUMN = "Outcome"

# Columns where a value of zero is biologically implausible and is therefore
# treated as a missing value marker in the Pima Indians dataset.
ZERO_AS_MISSING_COLUMNS = [
"Glucose",
"BloodPressure",
"SkinThickness",
"Insulin",
"BMI",
]

# Realistic validation ranges used for the manual prediction form.
INPUT_RANGES = {
"Pregnancies": (0, 20, 1),
"Glucose": (0, 300, 1),
"BloodPressure": (0, 200, 1),
"SkinThickness": (0, 100, 1),
"Insulin": (0, 900, 1),
"BMI": (0.0, 70.0, 0.1),
"DiabetesPedigreeFunction": (0.0, 3.0, 0.01),
"Age": (1, 120, 1),
}


# ---------------------------------------------------------------------------
# Data classes for organising results
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
"""Container holding a trained model and its evaluation results."""

name: str
model: object
y_pred: np.ndarray
y_proba: Optional[np.ndarray]
accuracy: float
precision: float
recall: float
f1: float
roc_auc: Optional[float]


@dataclass
class TrainingArtifacts:
"""Container holding everything produced by the training pipeline."""

results: Dict[str, ModelResult] = field(default_factory=dict)
best_model_name: str = ""
scaler: Optional[StandardScaler] = None
X_test: Optional[pd.DataFrame] = None
y_test: Optional[pd.Series] = None
feature_names: Optional[list] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(uploaded_file=None) -> Optional[pd.DataFrame]:
"""Load the Pima Indians Diabetes dataset.

Attempts to load `diabetes.csv` from the working directory first. If it
is not found, falls back to a user-uploaded file. Returns None if no
data source is available.

Args:
    uploaded_file: A file-like object from st.file_uploader, or None.

Returns:
    A pandas DataFrame containing the raw dataset, or None if
    no data source could be found.
"""
try:
    if os.path.exists(DATA_FILE_NAME):
        logger.info("Loading dataset from local file: %s", DATA_FILE_NAME)
        df = pd.read_csv(DATA_FILE_NAME)
        return df
    if uploaded_file is not None:
        logger.info("Loading dataset from user-uploaded file.")
        df = pd.read_csv(uploaded_file)
        return df
    logger.warning(
        "No dataset found: neither local file nor upload present.")
    return None
except (pd.errors.ParserError, FileNotFoundError, UnicodeDecodeError) as exc:
    logger.error("Failed to load dataset: %s", exc)
    st.error(f"Error loading dataset: {exc}")
    return None


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
"""Validate that the loaded dataframe matches the expected schema.

Args:
    df: The raw dataset.

Returns:
    A tuple of(is_valid, message).
"""
required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
missing_columns = required_columns - set(df.columns)
if missing_columns:
    return False, f"Dataset is missing required columns: {missing_columns}"
if df.empty:
    return False, "Dataset is empty."
if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
    return False, "Target column 'Outcome' must be binary (0 or 1)."
return True, "Dataset validated successfully."


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
"""Clean the raw dataset by handling implausible zero values.

Zero values in physiologically implausible columns(e.g. Glucose,
BloodPressure) are replaced with NaN, then imputed using the column
median. This is a standard preprocessing step for the Pima Indians
Diabetes dataset.

Args:
    df: Raw dataframe.

Returns:
    A cleaned copy of the dataframe.
"""
clean_df = df.copy()
for col in ZERO_AS_MISSING_COLUMNS:
    if col in clean_df.columns:
        zero_count = (clean_df[col] == 0).sum()
        if zero_count > 0:
            logger.info(
                "Column '%s': replacing %d zero values with median.", col, zero_count)
            clean_df[col] = clean_df[col].replace(0, np.nan)
            median_value = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(median_value)
clean_df = clean_df.drop_duplicates()
return clean_df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def build_model_zoo() -> Dict[str, object]:
"""Instantiate the seven classification models to be compared.

Returns:
    A dictionary mapping model name to an unfitted estimator.
"""
return {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}


@st.cache_resource(show_spinner=False)
def train_and_evaluate(df: pd.DataFrame) -> TrainingArtifacts:
"""Run the full training and evaluation pipeline.

Splits data, scales features, trains all seven models, evaluates each
on the held-out test set, and selects the best model based on F1-score.

Args:
    df: Cleaned dataframe.

Returns:
    A TrainingArtifacts object containing all results.
"""
logger.info("Starting training pipeline.")
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = build_model_zoo()
results: Dict[str, ModelResult] = {}

for name, model in models.items():
    try:
        logger.info("Training model: %s", name)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(
            y_test, y_proba) if y_proba is not None else None

        results[name] = ModelResult(
            name=name,
            model=model,
            y_pred=y_pred,
            y_proba=y_proba,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
        )
        logger.info("%s -> F1: %.4f | Accuracy: %.4f", name, f1, accuracy)
    except Exception as exc:  # noqa: BLE001 - log and continue with other models
        logger.error("Model '%s' failed to train: %s", name, exc)

if not results:
    raise RuntimeError(
        "All models failed to train. Check the dataset and logs.")

best_model_name = max(results, key=lambda k: results[k].f1)
logger.info("Best model selected by F1-score: %s", best_model_name)

artifacts = TrainingArtifacts(
    results=results,
    best_model_name=best_model_name,
    scaler=scaler,
    X_test=pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS),
    y_test=y_test.reset_index(drop=True),
    feature_names=FEATURE_COLUMNS,
)
return artifacts


def save_best_model(artifacts: TrainingArtifacts) -> None:
"""Persist the best model and the fitted scaler to disk.

Args:
    artifacts: The training artifacts produced by train_and_evaluate.
"""
try:
    best_model = artifacts.results[artifacts.best_model_name].model
    joblib.dump(best_model, MODEL_FILE_NAME)
    joblib.dump(artifacts.scaler, SCALER_FILE_NAME)
    logger.info("Saved best model ('%s') and scaler to disk.",
                artifacts.best_model_name)
except (OSError, joblib.externals.loky.process_executor.TerminatedWorkerError) as exc:
    logger.error("Failed to save model artifacts: %s", exc)
    st.warning(f"Could not save model to disk: {exc}")


def load_saved_model() -> Tuple[Optional[object], Optional[StandardScaler]]:
"""Load a previously saved model and scaler from disk, if present.

Returns:
    A tuple of(model, scaler), either of which may be None.
"""
model, scaler = None, None
try:
    if os.path.exists(MODEL_FILE_NAME):
        model = joblib.load(MODEL_FILE_NAME)
    if os.path.exists(SCALER_FILE_NAME):
        scaler = joblib.load(SCALER_FILE_NAME)
except (OSError, EOFError) as exc:
    logger.error("Failed to load saved model artifacts: %s", exc)
return model, scaler


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame) -> plt.Figure:
"""Plot the class distribution of the target variable."""
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=TARGET_COLUMN, data=df, ax=ax, palette="Set2")
ax.set_title("Class Distribution (0 = No Diabetes, 1 = Diabetes)")
ax.set_xlabel("Outcome")
ax.set_ylabel("Count")
fig.tight_layout()
return fig


def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
"""Plot histograms for each numeric feature."""
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, col in enumerate(FEATURE_COLUMNS):
    sns.histplot(df[col], kde=True, ax=axes[idx], color="steelblue")
    axes[idx].set_title(col)
fig.tight_layout()
return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
"""Plot a correlation heatmap for all numeric columns."""
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
fig.tight_layout()
return fig


def plot_confusion_matrix_fig(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> plt.Figure:
# Plot a confusion matrix for a given model's predictions.
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                              "No Diabetes", "Diabetes"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {model_name}")
fig.tight_layout()
return fig


def plot_roc_curve_fig(y_test: pd.Series, y_proba: np.ndarray, model_name: str) -> plt.Figure:
# Plot the ROC curve for a given model.
fig, ax = plt.subplots(figsize=(5, 4))
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_value = roc_auc_score(y_test, y_proba)
ax.plot(
    fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})", color="darkorange")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve - {model_name}")
ax.legend(loc="lower right")
fig.tight_layout()
return fig


def plot_feature_importance_fig(model: object, feature_names: list, model_name: str) -> Optional[plt.Figure]:
# Plot feature importance or coefficients for models that support it.
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
elif hasattr(model, "coef_"):
    importances = np.abs(model.coef_[0])
else:
    return None

order = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(
    [feature_names[i] for i in order][::-1],
    [importances[i] for i in order][::-1],
    color="teal",
)
ax.set_title(f"Feature Importance - {model_name}")
ax.set_xlabel("Relative Importance")
fig.tight_layout()
return fig


def plot_model_comparison(results: Dict[str, ModelResult]) -> plt.Figure:
# Plot a bar chart comparing F1-scores across all trained models.
names = list(results.keys())
f1_scores = [results[n].f1 for n in names]
order = np.argsort(f1_scores)[::-1]
sorted_names = [names[i] for i in order]
sorted_scores = [f1_scores[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2e7d32" if i == 0 else "#90a4ae" for i in range(len(sorted_names))]
ax.barh(sorted_names[::-1], sorted_scores[::-1], color=colors[::-1])
ax.set_xlabel("F1-Score")
ax.set_title("Model Comparison by F1-Score (best model highlighted)")
fig.tight_layout()
return fig


# ---------------------------------------------------------------------------
# Input validation for the manual prediction form
# ---------------------------------------------------------------------------
def validate_prediction_inputs(values: Dict[str, float]) -> Tuple[bool, str]:
# Validate manually entered patient data before prediction.
for feature, value in values.items():
    low, high, _ = INPUT_RANGES[feature]
    if value is None:
        return False, f"Missing value for '{feature}'."
    if value < low or value > high:
        return False, f"'{feature}' must be between {low} and {high}. Got {value}."
return True, ""


# ---------------------------------------------------------------------------
# Streamlit UI pages
# ---------------------------------------------------------------------------
def render_home_page() -> None:
# Render the Home / landing page.
st.title("🩺 Diabetes Prediction System")
st.markdown(
    """
    Welcome to the ** Diabetes Prediction System**, a machine-learning-powered
    application built on the ** Pima Indians Diabetes Dataset**.

    This system:
    - Explores and cleans the underlying clinical dataset
    - Trains and compares ** seven ** classification algorithms
    - Automatically selects the best-performing model using ** F1-score**
    - Lets you enter patient measurements to obtain a diabetes risk prediction

    Use the sidebar to navigate between the ** Exploratory Data Analysis**,
    **Model Comparison**, **Prediction**, and **About ** pages.
    """
)
st.info("Use the sidebar menu on the left to get started.")


def render_eda_page(df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
# Render the Exploratory Data Analysis page.
st.title("📊 Exploratory Data Analysis")

st.subheader("Raw Dataset Preview")
st.dataframe(df.head(10))

col1, col2 = st.columns(2)
with col1:
    st.metric("Number of Records", df.shape[0])
with col2:
    st.metric("Number of Features", len(FEATURE_COLUMNS))

st.subheader("Descriptive Statistics (After Cleaning)")
st.dataframe(clean_df.describe().T)

st.subheader("Class Balance")
st.pyplot(plot_class_balance(clean_df))

st.subheader("Feature Distributions")
st.pyplot(plot_feature_distributions(clean_df))

st.subheader("Correlation Heatmap")
st.pyplot(plot_correlation_heatmap(clean_df))


def render_model_comparison_page(artifacts: TrainingArtifacts) -> None:
# Render the model comparison and evaluation metrics page.
st.title("🤖 Model Training & Comparison")

st.markdown(
    "Seven classification models were trained and evaluated on a held-out "
    "test set (20% of the data). The table below summarises their performance."
)

metrics_rows = []
for name, result in artifacts.results.items():
    metrics_rows.append(
        {
            "Model": name,
            "Accuracy": round(result.accuracy, 4),
            "Precision": round(result.precision, 4),
            "Recall": round(result.recall, 4),
            "F1-Score": round(result.f1, 4),
            "ROC-AUC": round(result.roc_auc, 4) if result.roc_auc is not None else "N/A",
        }
    )
metrics_df = pd.DataFrame(metrics_rows).sort_values(
    "F1-Score", ascending=False)
st.dataframe(metrics_df, use_container_width=True)

st.success(
    f"✅ Best model selected automatically (highest F1-score): "
    f"**{artifacts.best_model_name}**"
)

st.subheader("Model Comparison Chart")
st.pyplot(plot_model_comparison(artifacts.results))

st.subheader("Detailed Evaluation for a Selected Model")
selected_model_name = st.selectbox(
    "Choose a model to inspect in detail:",
    options=list(artifacts.results.keys()),
    index=list(artifacts.results.keys()).index(artifacts.best_model_name),
)
selected_result = artifacts.results[selected_model_name]

col1, col2 = st.columns(2)
with col1:
    st.pyplot(
        plot_confusion_matrix_fig(
            artifacts.y_test, selected_result.y_pred, selected_model_name)
    )
with col2:
    if selected_result.y_proba is not None:
        st.pyplot(
            plot_roc_curve_fig(
                artifacts.y_test, selected_result.y_proba, selected_model_name)
        )
    else:
        st.warning(
            "ROC curve unavailable: model does not output probabilities.")

st.subheader("Feature Importance")
fi_fig = plot_feature_importance_fig(
    selected_result.model, artifacts.feature_names, selected_model_name
)
if fi_fig is not None:
    st.pyplot(fi_fig)
else:
    st.info(
        f"'{selected_model_name}' does not expose feature importances or coefficients.")


def render_prediction_page(artifacts: TrainingArtifacts) -> None:
# Render the manual patient-input prediction page.
st.title("🔮 Diabetes Risk Prediction")
st.markdown(
    f"Predictions are generated using the automatically selected best model: "
    f"**{artifacts.best_model_name}**"
)

with st.form("prediction_form"):
    st.subheader("Enter Patient Measurements")
    col1, col2 = st.columns(2)
    input_values = {}

    with col1:
        input_values["Pregnancies"] = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, step=1
        )
        input_values["Glucose"] = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=110, step=1
        )
        input_values["BloodPressure"] = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1
        )
        input_values["SkinThickness"] = st.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1
        )

    with col2:
        input_values["Insulin"] = st.number_input(
            "Insulin (mu U/mL)", min_value=0, max_value=900, value=80, step=1
        )
        input_values["BMI"] = st.number_input(
            "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1
        )
        input_values["DiabetesPedigreeFunction"] = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01
        )
        input_values["Age"] = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=30, step=1
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    is_valid, message = validate_prediction_inputs(input_values)
    if not is_valid:
        st.error(f"Input validation failed: {message}")
        logger.warning(
            "Prediction blocked due to invalid input: %s", message)
        return

    try:
        best_model = artifacts.results[artifacts.best_model_name].model
        input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]
        scaled_input = artifacts.scaler.transform(input_df)

        prediction = best_model.predict(scaled_input)[0]
        if hasattr(best_model, "predict_proba"):
            probability = best_model.predict_proba(scaled_input)[0][1]
        else:
            probability = None

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ The model predicts a HIGH risk of diabetes.")
        else:
            st.success("✅ The model predicts a LOW risk of diabetes.")

        if probability is not None:
            st.metric("Estimated Probability of Diabetes",
                      f"{probability * 100:.1f}%")

        st.caption(
            "This prediction is generated by a statistical model trained on "
            "historical data and is not a substitute for professional medical advice."
        )
        logger.info(
            "Prediction generated: class=%s, probability=%s",
            prediction,
            probability,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", exc)
        st.error(
            f"An error occurred while generating the prediction: {exc}")


def render_about_page() -> None:
# Render the About page with project and disclaimer information.
st.title("ℹ️ About This Application")
st.markdown(
    """
    **Diabetes Prediction System ** was developed as part of the * Advanced
    Machine Learning * module(COM763) portfolio assessment.

    **Dataset: ** Pima Indians Diabetes Dataset - 768 records of female
    patients of Pima Indian heritage, containing diagnostic measurements
    and a binary diabetes outcome.

    **Models compared: **
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine(SVM)
    - K-Nearest Neighbours(KNN)
    - Naive Bayes
    - Gradient Boosting

    **Model selection: ** The best-performing model is chosen automatically
    based on the highest ** F1-score ** on a held-out test set, balancing
    precision and recall for the positive(diabetic) class .

    **Disclaimer: ** This tool is for educational purposes only and must
    not be used for real clinical diagnosis or treatment decisions.
    """
)


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------
def main() -> None:
# Main entry point for the Streamlit application.
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Exploratory Data Analysis",
        "Model Training & Comparison", "Prediction", "About"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")

uploaded_file = None
if not os.path.exists(DATA_FILE_NAME):
    uploaded_file = st.sidebar.file_uploader(
        f"Upload '{DATA_FILE_NAME}' (not found in app directory)", type=["csv"]
    )

raw_df = load_data(uploaded_file)

if raw_df is None:
    st.warning(
        f"⚠️ No dataset available. Please add '{DATA_FILE_NAME}' to the app "
        "directory or upload it using the sidebar."
    )
    st.stop()

is_valid, message = validate_dataset(raw_df)
if not is_valid:
    st.error(f"Dataset validation failed: {message}")
    logger.error("Dataset validation failed: %s", message)
    st.stop()

clean_df = preprocess_data(raw_df)

try:
    artifacts = train_and_evaluate(clean_df)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
    return

if page == "Home":
    render_home_page()
elif page == "Exploratory Data Analysis":
    render_eda_page(raw_df, clean_df)
elif page == "Model Training & Comparison":
    render_model_comparison_page(artifacts)
    if st.sidebar.button("💾 Save Best Model to Disk"):
        save_best_model(artifacts)
        st.sidebar.success("Best model saved successfully.")
elif page == "Prediction":
    render_prediction_page(artifacts)
elif page == "About":
    render_about_page()


if __name__ == "__main__":
main()
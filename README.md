# CrediLens — Explainable Credit Risk Prediction & Decision Support System

CrediLens is an end-to-end Machine Learning decision support platform designed to estimate the probability of loan default on structured applicant data. The system features a production-ready **Streamlit** user interface, a **FastAPI** inference backend, and a tuned **XGBoost** model calibrated with Platt/Sigmoid scaling.

---

## 1. System Architecture

The application is structured into a demonstration and serving architecture:

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                   │
│                                                        │
│  [1. Risk Prediction]     [2. Model Evaluation]        │
│  [3. Explainability]      [4. Batch Scoring]           │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP JSON / In-Process Fallback
                            ▼
┌────────────────────────────────────────────────────────┐
│                    FastAPI Backend                     │
│                                                        │
│  GET  /health              GET  /model-info            │
│  POST /predict             POST /batch-predict         │
│  GET  /evaluation          GET  /explainability        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  ML Inference Service                  │
│                                                        │
│  - Preprocessing Pipeline (ColumnTransformer)          │
│  - Final Model (Tuned Calibrated XGBoost)             │
│  - 3 Baseline Comparison Models (LR, DT, RF)           │
│  - TreeSHAP Local & Global Explainer                  │
│  - Threshold Decision Engine (cutoff: 0.7431)          │
└────────────────────────────────────────────────────────┘
```

### Why FastAPI is Retained
Following the specifications in `docs/CrediLens_System_Architecture.md` and `docs/CrediLens_Frontend_Specification.md`:
1. **API Contracts**: Preserves clean service boundaries with standard RESTful contracts (`/predict`, `/batch-predict`, `/model-info`, `/evaluation`, `/explainability`).
2. **Decoupled Architecture**: Enables external systems (e.g., loan origination systems) to consume inferences independently of the frontend.
3. **Resilient Dual Mode**: The Streamlit frontend connects over HTTP to FastAPI when online, and automatically provides in-process model execution if Streamlit is launched standalone.

---

## 2. Core Functional Sections

### Section 1: Risk Prediction
- **Input Form**: Captures 11 applicant and loan features with interactive presets (*Prime Low-Risk*, *Borderline Moderate*, *High-Risk Distressed*).
- **Four-Model Observation**: Side-by-side comparison displaying default probabilities across **Logistic Regression**, **Decision Tree**, **Random Forest**, and the official **Calibrated XGBoost**.
- **Decision Support**: Displays recommendation (*Approve* vs *Deny*), operating threshold cutoff (0.7431), and risk tier categorization (*Low*, *Moderate*, *High*, *Critical*).
- **Local SHAP Breakdown**: Top feature attributions pushing risk up or pulling risk down.

### Section 2: Model Evaluation
- **Benchmark Metrics**: Comprehensive comparison table reporting **ROC-AUC**, **PR-AUC**, **Precision**, **Recall**, **F1-Score**, **Accuracy**, and **Brier Score**.
- **Diagnostic Curves**:
  - ROC Curves comparison for all 4 models.
  - Precision-Recall curves under class imbalance (~21.8% positive rate).
  - Calibration reliability curves (Uncalibrated vs Sigmoid Calibrated).
- **Confusion Matrix Explorer**: Interactive dropdown allowing evaluators to inspect True/False positives and negatives across all 4 architectures on the 6,305 held-out test records.
- **Selection Rationale**: Explains why Tuned XGBoost with Sigmoid Calibration was chosen over Random Forest and linear baselines.

### Section 3: Explainability
- **Global TreeSHAP Importance**: Ranked feature attributions across the portfolio.
- **Underwriting Insights**: Plain-language credit impact explanations for underwriters.
- **Interactive Simulator**: Dynamic what-if scenario simulator showing instant SHAP waterfall adjustments.
- **Attribution Notice**: Transparent disclaimer clarifying attribution vs causation.

### Section 4: Batch Prediction
- **Final Model Exclusivity**: In accordance with the system specification, batch scoring runs **exclusively on the champion Calibrated XGBoost** model.
- **CSV Portfolio Upload**: File validator and downloadable 10-applicant sample template.
- **Risk Stratification**: Portfolio default rate %, average risk, and tier distribution chart.
- **Export**: One-click download of scored results CSV.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- A clean virtual environment is recommended

### Step 1: Install Dependencies
A clean `requirements.txt` is provided, excluding all Jupyter and notebook-only packages:

```bash
pip install -r requirements.txt
```

### Step 2: Run Tests
Verify the complete test suite (backend API, inference, SHAP, and frontend integration):

```bash
python -m unittest discover -s tests
```

### Step 3: Start the Backend (Optional for standalone, recommended for full stack)
```bash
python main.py
# Or with uvicorn:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation is accessible at `http://127.0.0.1:8000/docs`.

### Step 4: Launch the Streamlit Frontend
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 4. Model & Artifact Details

| Property | Value |
| :--- | :--- |
| **Model Family** | XGBClassifier (tuned) + Sigmoid Calibration |
| **Estimator Pipeline** | Median Imputer + OneHotEncoder (`handle_unknown='ignore'`) + XGBoost |
| **Optimal Threshold** | `0.7431` (derived from training OOF PR-curve optimization) |
| **ROC-AUC** | `0.9471` (held-out test set: `0.9352`) |
| **PR-AUC** | `0.9043` (held-out test set: `0.8802`) |
| **Precision** | `96.83%` on default class |
| **Brier Score** | `0.0661` (calibrated) |
| **Held-out Test Size** | 6,305 applicants |

---

## 5. Directory Structure

```
Credlen/
├── app.py                      # Streamlit application entrypoint
├── main.py                     # FastAPI application entrypoint
├── requirements.txt            # Runtime dependencies (excludes Jupyter)
├── credit_risk_dataset.csv     # Source credit risk dataset
├── model/                      # Trained model directory
│   ├── Credit_risk_model.pkl   # Calibrated XGBoost pipeline
│   └── best_threshold.pkl      # Optimal decision threshold (0.7431)
├── backend/                    # FastAPI backend layer
│   ├── __init__.py
│   └── app/
│       ├── api.py              # REST API endpoints
│       ├── config.py           # Paths and server configs
│       ├── model_service.py    # Inference, 4 models, SHAP engine
│       └── schemas.py          # Pydantic request/response schemas
├── frontend/                   # Streamlit demonstration layer
│   ├── app.py                  # Main Streamlit layout & navigation
│   ├── styles.py               # Glassmorphism CSS design system
│   ├── api_client.py           # HTTP client with in-process fallback
│   └── views/
│       ├── prediction_view.py  # 1. Single risk prediction & 4-model comparison
│       ├── evaluation_view.py  # 2. Comprehensive benchmark evaluation & curves
│       ├── explainability_view.py # 3. TreeSHAP global & local attribution
│       └── batch_view.py       # 4. Final-model batch CSV scoring utility
├── tests/                      # Automated test suite
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_backend.py         # ModelService unit tests
│   └── test_frontend.py        # ApiClient & workflow tests
└── docs/                       # Project documentation & specifications
    ├── CrediLens_PRD.md
    ├── CrediLens_Frontend_Specification.md
    └── CrediLens_System_Architecture.md
```

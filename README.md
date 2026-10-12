# CrediLens — Explainable Credit Risk Prediction & Decision Support System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-34%20Passing-brightgreen.svg)]()
[![Code Quality](https://img.shields.io/badge/Stability-Production--Hardened-success.svg)]()

**CrediLens** is an end-to-end Machine Learning decision support platform designed to estimate the probability of loan default on structured applicant data. The system combines a modern **Streamlit** Glassmorphism dashboard, a high-throughput **FastAPI** inference backend, and a tuned **XGBoost** model calibrated with Platt/Sigmoid scaling for mathematically sound, well-calibrated risk scoring.

---

## 1. System Architecture

CrediLens is structured into a modular, production-ready serving architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend                        │
│                                                                 │
│  [1. Risk Prediction]           [2. Model Evaluation]           │
│  [3. Explainability & SHAP]     [4. Batch CSV Scoring]          │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP JSON / In-Process Fallback
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                         │
│                                                                 │
│  GET  /health                   GET  /model-info                │
│  POST /predict                  POST /batch-predict             │
│  POST /predict-batch (alias)    GET  /explainability            │
│  GET  /evaluation                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML Inference Engine                        │
│                                                                 │
│  - Preprocessing Pipeline (ColumnTransformer, OneHot, Imputer)  │
│  - Champion Model (Tuned Sigmoid-Calibrated XGBoost)            │
│  - 3 Baseline Comparison Models (LR, DT, Random Forest)         │
│  - TreeSHAP Feature Attributions (Local Waterfall & Global)     │
│  - Threshold Decision Engine (Operating Cutoff: 0.7632)         │
└─────────────────────────────────────────────────────────────────┘
```

### Architectural Key Principles
1. **Strict RESTful API Contracts**: Standardized Pydantic v2 schemas across all inference, evaluation, and system endpoints.
2. **Dual-Route Batch Support**: Exposes both `/batch-predict` and `/predict-batch` aliases to ensure broad client compatibility.
3. **Resilient Dual Mode**: The Streamlit frontend connects over HTTP to the FastAPI service when online, and automatically falls back to direct in-process inference if run standalone.
4. **Reactive Frontend**: The root entrypoint (`app.py`) delegates cleanly to `app.frontend.streamlit_app.main()`, ensuring complete state preservation and re-rendering across all user interactions without blank-screen regressions.

---

## 2. Core Functional Modules

### 🎯 1. Risk Prediction
- **Interactive Form**: Captures 11 applicant and loan attributes with instant input validation.
- **One-Click Presets**: Pre-populated profiles (*Prime Low-Risk*, *Borderline Moderate-Risk*, *High-Risk Solvency Distressed*).
- **Four-Model Observation**: Side-by-side comparison displaying default probabilities across **Logistic Regression**, **Decision Tree**, **Random Forest**, and the official **Calibrated XGBoost**.
- **Underwriting Decision Card**: Displays recommendation (*Approve* vs *Deny*), operating threshold cutoff (0.7632), certainty buffer, and risk tier categorization (*Low*, *Moderate*, *High*, *Critical*).
- **Local SHAP Breakdown**: Dynamic attribution listing top risk-elevating (red) and risk-mitigating (green) feature drivers.

### 📊 2. Model Evaluation & Benchmarks
- **Benchmark Metrics Table**: Comprehensive multi-model evaluation across held-out test data (6,305 applicants).
- **Multi-Metric Comparative Chart**: Grouped Altair bar charts comparing ROC-AUC, PR-AUC, Precision, Recall, and F1-Score across all 4 architectures.
- **Diagnostic Curve Tabs**:
  - **ROC Curves**: True Positive Rate vs False Positive Rate for all models.
  - **Precision-Recall Curves**: PR-AUC under dataset class imbalance (~21.8% default rate).
  - **Calibration Reliability**: Uncalibrated vs Sigmoid Calibrated reliability curves and empirical Brier scores.
- **Model Selection Rationale**: Clear, auditable explanation of why Calibrated XGBoost was selected over baseline alternatives.

### 🔍 3. Explainability & SHAP Insights
- **Global TreeSHAP Importance**: Ranked relative impact weights across the portfolio.
- **Underwriting Insights Directory**: Plain-language risk interpretations explaining financial mechanics (e.g. debt-to-income impact, interest rate burden).
- **Interactive What-If Simulator**: Real-time slider simulator showing how adjusting income, loan amount, interest rate, or tenure shifts the model's prediction.

### 📁 4. Batch Prediction Utility
- **High-Throughput Scoring**: Evaluates multi-applicant CSV portfolios exclusively using the champion Calibrated XGBoost model.
- **Sample Template**: Downloadable 10-applicant sample CSV template.
- **Quick Test Runner**: Instant one-click evaluation of 10 diverse test applicants.
- **Portfolio Analytics**: Aggregated default rate %, average portfolio risk score, and risk tier distribution bar chart.
- **Export**: One-click download of scored results with predicted probabilities, decisions, and risk bands.

---

## 3. Benchmark Model Performance

Evaluation on 6,305 held-out test records:

| Model Architecture | Model Tier | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned + Calibrated)** | **Champion** | **0.9471** | **0.9043** | **96.83%** | 69.46% | **80.89%** | **0.0661** |
| Random Forest | Baseline | 0.9284 | 0.8803 | 96.56% | 71.72% | 82.30% | 0.0820 |
| Decision Tree | Baseline | 0.8417 | 0.6052 | 72.66% | 76.24% | 74.40% | 0.1583 |
| Logistic Regression | Baseline | 0.8705 | 0.7135 | 54.48% | 77.81% | 64.08% | 0.1245 |

> **Key Insight**: The Sigmoid-Calibrated XGBoost model achieves the lowest Brier Score (`0.0661`) and highest Precision (`96.83%`), minimizing false approvals while outputting true, reliable default probabilities.

---

## 4. Applicant Feature Schema

The system validates 11 core financial and demographic features:

| Field Name | Type | Valid Range / Categories | Description |
| :--- | :---: | :---: | :--- |
| `person_age` | Integer | 18 – 100 | Applicant age in years |
| `person_income` | Float | $1,000 – $2,000,000 | Annual income in USD |
| `person_home_ownership` | String | `RENT`, `MORTGAGE`, `OWN`, `OTHER` | Housing status |
| `person_emp_length` | Float | 0.0 – 60.0 | Employment tenure in years |
| `loan_intent` | String | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` | Purpose of requested loan |
| `loan_grade` | String | `A`, `B`, `C`, `D`, `E`, `F`, `G` | Risk grade assigned by credit bureau |
| `loan_amnt` | Float | $500 – $50,000 | Requested principal amount |
| `loan_int_rate` | Float | 1.0% – 40.0% | Annual percentage interest rate |
| `loan_percent_income` | Float | 0.0 – 1.5 | Debt-to-income ratio (`loan_amnt / income`) |
| `cb_person_default_on_file` | String | `Y`, `N` | Historical default flag on bureau record |
| `cb_person_cred_hist_length`| Integer | 0 – 50 | Length of credit history in years |

---

## 5. Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested through Python 3.14)
- Git

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/dheerajrai-dev/Credlen.git
cd Credlen

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Test Suite
Verify that all 34 automated unit, integration, and stress tests pass:
```bash
python -m unittest discover tests -v
```

### Step 4: Launch the FastAPI Backend
```bash
python main.py
```
- Server starts at `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Redoc Documentation: `http://127.0.0.1:8000/redoc`

### Step 5: Launch the Streamlit Frontend
In a new terminal window:
```bash
streamlit run app.py
```
- Open `http://localhost:8501` in your browser.

---

## 6. API Reference & Usage Examples

### 1. Single Loan Risk Scoring (`POST /predict`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "person_age": 36,
       "person_income": 82000.0,
       "person_home_ownership": "MORTGAGE",
       "person_emp_length": 8.0,
       "loan_intent": "HOMEIMPROVEMENT",
       "loan_grade": "A",
       "loan_amnt": 10000.0,
       "loan_int_rate": 7.49,
       "loan_percent_income": 0.12,
       "cb_person_default_on_file": "N",
       "cb_person_cred_hist_length": 10
     }'
```

**Response (HTTP 200)**:
```json
{
  "applicant_data": { ... },
  "predictions": {
    "logistic_regression": 0.1478,
    "decision_tree": 0.0,
    "random_forest": 0.021,
    "xgboost": 0.0159
  },
  "final_model": {
    "name": "Calibrated XGBoost (Sigmoid)",
    "probability": 0.0159,
    "threshold": 0.7632,
    "decision": "Repaid",
    "risk_band": "Low"
  },
  "top_factors": [
    {
      "feature_name": "loan_percent_income",
      "contribution": -0.842,
      "direction": "decreases_risk",
      "description": "Favorable debt-to-income ratio lowers default likelihood."
    }
  ]
}
```

### 2. Batch Portfolio Scoring (`POST /batch-predict` or `POST /predict-batch`)
```bash
curl -X POST "http://127.0.0.1:8000/batch-predict" \
     -F "file=@credit_risk_dataset.csv"
```

**Response (HTTP 200)**:
```json
{
  "total_records": 10,
  "predicted_defaults": 5,
  "predicted_repays": 5,
  "default_rate": 50.0,
  "average_risk": 0.505,
  "results": [ ... ]
}
```

### 3. Service Health & Readiness (`GET /health`)
```bash
curl "http://127.0.0.1:8000/health"
```

**Response (HTTP 200)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Calibrated XGBoost (Sigmoid)",
  "threshold": 0.7632,
  "uptime_seconds": 124.5
}
```

---

## 7. Directory Structure

```text
Credlen/
├── app/
│   ├── api/
│   │   ├── main.py                 # FastAPI application, CORS & root routes
│   │   ├── lifespan.py             # Model lifecycle manager (startup/shutdown)
│   │   ├── dependencies.py         # Request state dependency injection
│   │   └── routes/
│   │       └── prediction.py       # Prediction, evaluation & dual batch routes
│   ├── frontend/
│   │   ├── streamlit_app.py        # Streamlit layout, navigation & main() entrypoint
│   │   ├── api_client.py           # HTTP client with in-process fallback
│   │   ├── styles.py               # Glassmorphism CSS design system
│   │   └── views/
│   │       ├── prediction_view.py  # 1. Single risk prediction & 4-model comparison
│   │       ├── evaluation_view.py  # 2. Comprehensive benchmark evaluation & curves
│   │       ├── explainability_view.py # 3. TreeSHAP global & local attribution
│   │       └── batch_view.py       # 4. Final-model batch CSV scoring utility
│   ├── services/
│   │   └── prediction_service.py   # Prediction & explainability engine
│   ├── schemas/
│   │   └── credit.py               # Pydantic request/response schemas
│   └── core/
│       ├── config.py               # Dynamic path resolution & configs
│       └── exceptions.py           # Domain-specific custom exceptions
├── models/                         # Serialized ML artifacts
│   ├── Credit_risk_model.pkl       # Calibrated XGBoost pipeline
│   ├── best_threshold.pkl          # Optimal decision threshold (0.7632)
│   └── baselines.joblib            # Pre-cached baseline models (LR, DT, RF)
├── tests/                          # 34 automated regression & stress tests
│   ├── test_api.py                 # FastAPI endpoint & lifespan tests
│   ├── test_frontend.py            # ApiClient & Streamlit entrypoint tests
│   ├── test_services.py            # PredictionService unit tests
│   ├── test_m2_adversarial.py      # Boundary values & malformed payload tests
│   └── test_stress_batch.py        # 500-row stress, empty & corrupted CSV tests
├── docs/                           # Architecture, PRD & design specifications
├── requirements.txt                # Production runtime dependencies
├── .env.example                    # Environment configuration template
├── .gitignore                      # Git exclusion rules
├── README.md                       # Complete documentation
├── main.py                         # Root entrypoint for FastAPI backend
└── app.py                          # Root entrypoint for Streamlit frontend
```

---

## 8. Automated Test Suite (34 Tests)

The test suite covers full regression, boundary conditions, and adversarial inputs:

```bash
python -m unittest discover tests -v
```

| Test Module | Tests | Focus Areas |
| :--- | :---: | :--- |
| `test_api.py` | 12 | Endpoints (`/`, `/health`, `/model-info`, `/predict`, `/batch-predict`, `/predict-batch`, `/evaluation`, `/explainability`), boundary 422 validations, CSV extension handling. |
| `test_frontend.py` | 5 | `ApiClient` HTTP communications, health probes, predict/batch workflows, and verifies that `streamlit_app` exposes a callable `main()`. |
| `test_services.py` | 5 | ML engine initialization, single scoring, batch scoring, baseline comparisons, and TreeSHAP calculations. |
| `test_m2_adversarial.py` | 4 | All 16 boundary combinations of applicant inputs, extreme ages, maximum incomes, and malformed payload bodies. |
| `test_stress_batch.py` | 8 | 500-row batch stress tests, 0-byte files, header-only CSVs, missing columns, non-numeric values, and corrupted extensions. |

**Result**: `34 passed in ~9.5s (100% pass rate, 0 failures, 0 errors)`.

---

## 9. License & Attribution

Developed as an academic Machine Learning portfolio demonstrating production-grade software engineering, explainable artificial intelligence (TreeSHAP), and calibrated risk underwriting.

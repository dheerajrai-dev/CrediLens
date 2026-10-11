from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

class ApplicantInput(BaseModel):
    person_age: int = Field(default=25, ge=18, le=100, description="Applicant age")
    person_income: float = Field(default=50000.0, ge=0.0, description="Annual income in USD")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        default="RENT", description="Home ownership status"
    )
    person_emp_length: float = Field(
        default=3.0, ge=0.0, le=60.0, description="Employment length in years"
    )
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ] = Field(default="PERSONAL", description="Loan intent")
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        default="B", description="Assigned credit grade"
    )
    loan_amnt: float = Field(default=10000.0, ge=500.0, le=50000.0, description="Loan amount")
    loan_int_rate: float = Field(
        default=11.5, ge=1.0, le=40.0, description="Loan interest rate (%)"
    )
    loan_percent_income: Optional[float] = Field(
        default=None, ge=0.0, le=1.5, description="Loan amount as percentage of income"
    )
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        default="N", description="Historical default on file"
    )
    cb_person_cred_hist_length: int = Field(
        default=3, ge=0, le=50, description="Credit history length in years"
    )

class FactorContribution(BaseModel):
    feature: str
    feature_name: str
    value: Any
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    description: str

class FinalModelResult(BaseModel):
    name: str
    probability: float
    threshold: float
    prediction: int
    decision: str
    risk_band: str

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    final_model: FinalModelResult
    top_factors: List[FactorContribution]
    applicant_data: Dict[str, Any]

class BatchItemResult(BaseModel):
    applicant_id: int
    default_probability: float
    prediction: int
    decision: str
    risk_band: str

class BatchPredictionResponse(BaseModel):
    total_records: int
    predicted_defaults: int
    predicted_repays: int
    default_rate: float
    average_risk: float
    results: List[BatchItemResult]

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    dataset: str
    target: str
    threshold: float
    feature_columns: Dict[str, List[str]]
    train_size: int
    test_size: int
    training_date: str
    library_versions: Dict[str, str]
    metrics: Dict[str, float]

class EvaluationMetrics(BaseModel):
    model: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    brier_score: float

class ConfusionMatrixData(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    precision: float
    recall: float
    f1: float

class EvaluationResponse(BaseModel):
    comparison_table: List[EvaluationMetrics]
    curves: Dict[str, Any]
    confusion_matrices: Dict[str, ConfusionMatrixData]
    selected_model: str
    selection_rationale: str

class GlobalFeatureImportance(BaseModel):
    feature: str
    label: str
    importance: float
    mean_abs_shap: float
    trend: str
    business_impact: str

class ExplainabilityResponse(BaseModel):
    model_name: str
    global_importance: List[GlobalFeatureImportance]
    methodology: str
    disclaimer: str

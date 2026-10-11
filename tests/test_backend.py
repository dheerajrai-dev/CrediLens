import unittest
from backend.app.schemas import ApplicantInput
from backend.app.model_service import ModelService
import pandas as pd

class TestModelService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ModelService.get_instance()

    def test_health_and_loaded(self):
        self.assertTrue(self.service.is_loaded)
        self.assertIsNotNone(self.service.final_model)
        self.assertGreater(self.service.threshold, 0.0)

    def test_single_prediction(self):
        applicant = ApplicantInput(
            person_age=30,
            person_income=65000.0,
            person_home_ownership="MORTGAGE",
            person_emp_length=5.0,
            loan_intent="EDUCATION",
            loan_grade="A",
            loan_amnt=8000.0,
            loan_int_rate=7.5,
            loan_percent_income=0.12,
            cb_person_default_on_file="N",
            cb_person_cred_hist_length=6
        )
        resp = self.service.predict_single(applicant)
        self.assertIn("xgboost", resp.predictions)
        self.assertIn("logistic_regression", resp.predictions)
        self.assertIn("decision_tree", resp.predictions)
        self.assertIn("random_forest", resp.predictions)
        self.assertGreaterEqual(resp.final_model.probability, 0.0)
        self.assertLessEqual(resp.final_model.probability, 1.0)
        self.assertIn(resp.final_model.risk_band, ["Low", "Moderate", "High", "Critical"])
        self.assertTrue(len(resp.top_factors) > 0)

    def test_batch_prediction(self):
        df = pd.DataFrame([
            {
                "person_age": 22,
                "person_income": 25000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 1.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "D",
                "loan_amnt": 12000.0,
                "loan_int_rate": 16.5,
                "loan_percent_income": 0.48,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 2
            },
            {
                "person_age": 45,
                "person_income": 120000.0,
                "person_home_ownership": "OWN",
                "person_emp_length": 15.0,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "A",
                "loan_amnt": 15000.0,
                "loan_int_rate": 6.2,
                "loan_percent_income": 0.12,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 14
            }
        ])
        batch_resp = self.service.predict_batch(df)
        self.assertEqual(batch_resp.total_records, 2)
        self.assertEqual(len(batch_resp.results), 2)
        # High risk borrower should have higher probability than low risk borrower
        p1 = batch_resp.results[0].default_probability
        p2 = batch_resp.results[1].default_probability
        self.assertGreater(p1, p2)

    def test_evaluation_data(self):
        eval_data = self.service.get_evaluation_data()
        self.assertEqual(len(eval_data.comparison_table), 4)
        self.assertIn("XGBoost (Tuned + Calibrated)", [m.model for m in eval_data.comparison_table])
        self.assertIn("Logistic Regression", [m.model for m in eval_data.comparison_table])
        self.assertIn("roc", eval_data.curves)
        self.assertIn("pr", eval_data.curves)
        self.assertIn("calibration", eval_data.curves)

    def test_explainability_data(self):
        expl = self.service.get_explainability_data()
        self.assertTrue(len(expl.global_importance) >= 5)
        self.assertTrue(any("loan_percent_income" in f.feature for f in expl.global_importance))

if __name__ == "__main__":
    unittest.main()

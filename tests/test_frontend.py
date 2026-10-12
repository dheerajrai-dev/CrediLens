import unittest
import pandas as pd
from app.frontend.api_client import ApiClient

class TestFrontendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = ApiClient()

    def test_client_health(self):
        health = self.client.get_health()
        self.assertIn("status", health)
        self.assertTrue(health["model_loaded"])

    def test_client_predict(self):
        applicant = {
            "person_age": 32,
            "person_income": 72000.0,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 6.0,
            "loan_intent": "VENTURE",
            "loan_grade": "A",
            "loan_amnt": 10000.0,
            "loan_int_rate": 8.0,
            "loan_percent_income": 0.14,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 8
        }
        res = self.client.predict(applicant)
        self.assertIn("final_model", res)
        self.assertIn("probability", res["final_model"])
        self.assertIn("predictions", res)
        self.assertEqual(len(res["predictions"]), 4)

    def test_client_batch(self):
        sample_df = pd.DataFrame([
            {
                "person_age": 25,
                "person_income": 40000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 2.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 8000.0,
                "loan_int_rate": 11.0,
                "loan_percent_income": 0.20,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3
            }
        ])
        batch_res = self.client.predict_batch(sample_df)
        self.assertEqual(batch_res["total_records"], 1)
        self.assertEqual(len(batch_res["results"]), 1)

    def test_client_evaluation_and_explainability(self):
        ev = self.client.get_evaluation()
        self.assertEqual(len(ev["comparison_table"]), 4)

        ex = self.client.get_explainability()
        self.assertTrue(len(ex["global_importance"]) > 0)
    def test_streamlit_app_entrypoint_callable(self):
        import app.frontend.streamlit_app as st_app
        self.assertTrue(hasattr(st_app, "main"))
        self.assertTrue(callable(st_app.main))

if __name__ == "__main__":
    unittest.main()

import unittest
from starlette.testclient import TestClient
from backend.app.api import app

class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_health(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("CrediLens", resp.json()["project"])

        h_resp = self.client.get("/health")
        self.assertEqual(h_resp.status_code, 200)
        self.assertTrue(h_resp.json()["model_loaded"])

    def test_model_info(self):
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["target"], "loan_status")
        self.assertIn("roc_auc", data["metrics"])

    def test_predict_endpoint(self):
        payload = {
            "person_age": 28,
            "person_income": 58000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 8000,
            "loan_int_rate": 10.5,
            "loan_percent_income": 0.14,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 4
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("xgboost", data["predictions"])
        self.assertIn("probability", data["final_model"])
        self.assertIn("decision", data["final_model"])

    def test_evaluation_endpoint(self):
        resp = self.client.get("/evaluation")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["comparison_table"]), 4)

    def test_explainability_endpoint(self):
        resp = self.client.get("/explainability")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(len(data["global_importance"]) > 0)

if __name__ == "__main__":
    unittest.main()

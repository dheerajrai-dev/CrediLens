import io
import unittest
import warnings

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except ImportError:
    pass

from fastapi.testclient import TestClient
from app.api.main import app

class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

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

    VALID_CSV = (
        "person_age,person_income,person_home_ownership,person_emp_length,"
        "loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,"
        "cb_person_default_on_file,cb_person_cred_hist_length\n"
        "28,58000,RENT,4.0,PERSONAL,B,8000,10.5,0.14,N,4\n"
        "35,80000,MORTGAGE,10.0,EDUCATION,A,15000,8.5,0.19,N,7\n"
    )

    def test_batch_predict_endpoint_valid_csv(self):
        resp = self.client.post(
            "/batch-predict",
            files={"file": ("batch.csv", io.BytesIO(self.VALID_CSV.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_records"], 2)
        self.assertIn("predicted_defaults", data)
        self.assertIn("predicted_repays", data)
        self.assertIn("default_rate", data)
        self.assertIn("average_risk", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        first_item = data["results"][0]
        self.assertIn("applicant_id", first_item)
        self.assertIn("default_probability", first_item)
        self.assertIn("prediction", first_item)
        self.assertIn("decision", first_item)
        self.assertIn("risk_band", first_item)

    def test_predict_batch_endpoint_alias(self):
        resp = self.client.post(
            "/predict-batch",
            files={"file": ("batch_alias.csv", io.BytesIO(self.VALID_CSV.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_records"], 2)
        self.assertIn("predicted_defaults", data)
        self.assertIn("predicted_repays", data)
        self.assertIn("default_rate", data)
        self.assertIn("average_risk", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        first_item = data["results"][0]
        self.assertIn("applicant_id", first_item)
        self.assertIn("default_probability", first_item)
        self.assertIn("prediction", first_item)
        self.assertIn("decision", first_item)
        self.assertIn("risk_band", first_item)

    def test_batch_predict_extensions(self):
        # .CSV and .txt return 200
        resp_csv = self.client.post(
            "/batch-predict",
            files={"file": ("batch_data.CSV", io.BytesIO(self.VALID_CSV.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(resp_csv.status_code, 200)

        resp_txt = self.client.post(
            "/batch-predict",
            files={"file": ("batch_data.txt", io.BytesIO(self.VALID_CSV.encode("utf-8")), "text/plain")},
        )
        self.assertEqual(resp_txt.status_code, 200)

        # .json and .bin return 400
        resp_json = self.client.post(
            "/batch-predict",
            files={"file": ("batch_data.json", io.BytesIO(b'{"key": "value"}'), "application/json")},
        )
        self.assertEqual(resp_json.status_code, 400)

        resp_bin = self.client.post(
            "/batch-predict",
            files={"file": ("batch_data.bin", io.BytesIO(b"\x00\x01\x02\x03"), "application/octet-stream")},
        )
        self.assertEqual(resp_bin.status_code, 400)

    def test_batch_predict_empty_file(self):
        resp = self.client.post(
            "/batch-predict",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        self.assertEqual(resp.status_code, 422)

    def test_batch_predict_missing_columns(self):
        incomplete_csv = "person_age,person_income\n28,58000\n"
        resp = self.client.post(
            "/batch-predict",
            files={"file": ("missing_cols.csv", io.BytesIO(incomplete_csv.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(resp.status_code, 422)

    def test_predict_single_validation_errors(self):
        base_payload = {
            "person_age": 28,
            "person_income": 58000.0,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 8000.0,
            "loan_int_rate": 10.5,
            "loan_percent_income": 0.14,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 4,
        }

        # negative income (-500)
        p = base_payload.copy()
        p["person_income"] = -500.0
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

        # age under 18 (15)
        p = base_payload.copy()
        p["person_age"] = 15
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

        # invalid loan_intent ("GAMBLING")
        p = base_payload.copy()
        p["loan_intent"] = "GAMBLING"
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

        # invalid loan_grade ("Z")
        p = base_payload.copy()
        p["loan_grade"] = "Z"
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

        # loan_amnt out of bounds (100.0)
        p = base_payload.copy()
        p["loan_amnt"] = 100.0
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

        # loan_amnt out of bounds (100000.0)
        p = base_payload.copy()
        p["loan_amnt"] = 100000.0
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

        # negative loan_int_rate (-1.0)
        p = base_payload.copy()
        p["loan_int_rate"] = -1.0
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 422)

    def test_predict_single_boundary_values(self):
        base_payload = {
            "person_age": 28,
            "person_income": 58000.0,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 8000.0,
            "loan_int_rate": 10.5,
            "loan_percent_income": 0.14,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 4,
        }

        # min loan amnt (500.0)
        p = base_payload.copy()
        p["loan_amnt"] = 500.0
        p["loan_percent_income"] = round(500.0 / p["person_income"], 2)
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 200)

        # max loan amnt (50000.0)
        p = base_payload.copy()
        p["loan_amnt"] = 50000.0
        p["loan_percent_income"] = round(50000.0 / p["person_income"], 2)
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 200)

        # min age (18)
        p = base_payload.copy()
        p["person_age"] = 18
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 200)

        # max age (100)
        p = base_payload.copy()
        p["person_age"] = 100
        resp = self.client.post("/predict", json=p)
        self.assertEqual(resp.status_code, 200)

if __name__ == "__main__":
    unittest.main()

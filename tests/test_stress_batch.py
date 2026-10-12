import io
import time
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.schemas.credit import BatchPredictionResponse

class TestStressBatchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def _generate_valid_df(self, n_rows=10):
        np.random.seed(42)
        home_options = ["RENT", "OWN", "MORTGAGE", "OTHER"]
        intent_options = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        grade_options = ["A", "B", "C", "D", "E", "F", "G"]
        default_options = ["Y", "N"]

        records = []
        for i in range(n_rows):
            age = int(np.random.randint(18, 70))
            income = float(np.random.randint(20000, 150000))
            home = home_options[i % len(home_options)]
            emp_len = float(np.random.randint(0, 20))
            intent = intent_options[i % len(intent_options)]
            grade = grade_options[i % len(grade_options)]
            loan_amnt = float(np.random.randint(1000, 35000))
            int_rate = float(round(np.random.uniform(5.0, 22.0), 2))
            lpi = round(min(loan_amnt / income, 1.5), 2)
            default = default_options[i % len(default_options)]
            cred_hist = int(np.random.randint(1, 15))

            records.append({
                "person_age": age,
                "person_income": income,
                "person_home_ownership": home,
                "person_emp_length": emp_len,
                "loan_intent": intent,
                "loan_grade": grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": int_rate,
                "loan_percent_income": lpi,
                "cb_person_default_on_file": default,
                "cb_person_cred_hist_length": cred_hist,
            })
        return pd.DataFrame(records)

    def _df_to_csv_bytes(self, df):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    def test_valid_batch_both_endpoints_and_schema_conformity(self):
        """Test both /batch-predict and /predict-batch on valid data and assert exact schema."""
        df = self._generate_valid_df(15)
        csv_bytes = self._df_to_csv_bytes(df)

        for endpoint in ["/batch-predict", "/predict-batch"]:
            with self.subTest(endpoint=endpoint):
                resp = self.client.post(
                    endpoint,
                    files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
                )
                self.assertEqual(resp.status_code, 200, f"Failed on {endpoint}: {resp.text}")
                data = resp.json()

                # Validate with Pydantic model directly
                validated = BatchPredictionResponse.model_validate(data)
                self.assertEqual(validated.total_records, 15)
                self.assertEqual(len(validated.results), 15)
                self.assertEqual(validated.predicted_defaults + validated.predicted_repays, 15)

                calc_defaults = sum(1 for r in validated.results if r.prediction == 1)
                self.assertEqual(validated.predicted_defaults, calc_defaults)
                expected_default_rate = round(calc_defaults / 15, 4)
                self.assertEqual(validated.default_rate, expected_default_rate)

                mean_risk = round(float(np.mean([r.default_probability for r in validated.results])), 4)
                self.assertAlmostEqual(validated.average_risk, mean_risk, places=4)

                for idx, item in enumerate(validated.results):
                    self.assertEqual(item.applicant_id, idx + 1)
                    self.assertTrue(0.0 <= item.default_probability <= 1.0)
                    self.assertIn(item.prediction, [0, 1])
                    self.assertEqual(item.decision, "Default" if item.prediction == 1 else "Repaid")
                    self.assertIn(item.risk_band, ["Low", "Moderate", "High", "Critical"])

    def test_large_batch_500_rows_both_endpoints(self):
        """Stress-test batch prediction with 500 rows on both endpoints."""
        df_500 = self._generate_valid_df(500)
        csv_bytes = self._df_to_csv_bytes(df_500)

        for endpoint in ["/batch-predict", "/predict-batch"]:
            with self.subTest(endpoint=endpoint):
                start = time.perf_counter()
                resp = self.client.post(
                    endpoint,
                    files={"file": ("large_500.csv", io.BytesIO(csv_bytes), "text/csv")},
                )
                duration = time.perf_counter() - start

                self.assertEqual(resp.status_code, 200, f"500-row batch failed on {endpoint}: {resp.text}")
                data = resp.json()
                validated = BatchPredictionResponse.model_validate(data)
                self.assertEqual(validated.total_records, 500)
                self.assertEqual(len(validated.results), 500)
                self.assertEqual(validated.predicted_defaults + validated.predicted_repays, 500)
                self.assertLess(duration, 10.0, "Batch processing 500 rows should complete within 10 seconds")

    def test_corrupted_empty_file_zero_bytes(self):
        """Empty 0-byte file should return 422 with meaningful error on both endpoints."""
        for endpoint in ["/batch-predict", "/predict-batch"]:
            with self.subTest(endpoint=endpoint):
                resp = self.client.post(
                    endpoint,
                    files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
                )
                self.assertEqual(resp.status_code, 422, f"Endpoint {endpoint} gave status {resp.status_code}")
                self.assertIn("detail", resp.json())

    def test_corrupted_header_only_zero_rows(self):
        """CSV with header row but 0 records should return 200 with total_records=0."""
        header_csv = (
            "person_age,person_income,person_home_ownership,person_emp_length,"
            "loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,"
            "cb_person_default_on_file,cb_person_cred_hist_length\n"
        )
        resp = self.client.post(
            "/batch-predict",
            files={"file": ("header_only.csv", io.BytesIO(header_csv.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        validated = BatchPredictionResponse.model_validate(data)
        self.assertEqual(validated.total_records, 0)
        self.assertEqual(len(validated.results), 0)

    def test_corrupted_missing_required_columns(self):
        """Missing key columns should return 422 with descriptive error naming the missing column."""
        cols = [
            "person_age", "person_income", "person_home_ownership", "person_emp_length",
            "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
            "cb_person_default_on_file", "cb_person_cred_hist_length"
        ]
        df_full = self._generate_valid_df(5)

        for col in cols:
            with self.subTest(missing_col=col):
                df_missing = df_full.drop(columns=[col])
                csv_bytes = self._df_to_csv_bytes(df_missing)
                resp = self.client.post(
                    "/batch-predict",
                    files={"file": (f"missing_{col}.csv", io.BytesIO(csv_bytes), "text/csv")},
                )
                self.assertEqual(resp.status_code, 422, f"Failed for missing col {col}: got {resp.status_code}")
                self.assertIn(col, resp.json()["detail"])

    def test_corrupted_non_numeric_values_in_numeric_columns(self):
        """Non-numeric values in numeric columns: server must not crash with 500."""
        corrupt_csv = (
            "person_age,person_income,person_home_ownership,person_emp_length,"
            "loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,"
            "cb_person_default_on_file,cb_person_cred_hist_length\n"
            "invalid_age,58000,RENT,4.0,PERSONAL,B,8000,10.5,0.14,N,4\n"
            "28,invalid_income,RENT,4.0,PERSONAL,B,8000,10.5,0.14,N,4\n"
            "28,58000,RENT,4.0,PERSONAL,B,invalid_amount,10.5,0.14,N,4\n"
        )
        resp = self.client.post(
            "/batch-predict",
            files={"file": ("corrupt_numeric.csv", io.BytesIO(corrupt_csv.encode("utf-8")), "text/csv")},
        )
        self.assertIn(resp.status_code, [200, 422], f"Server crashed or returned unexpected {resp.status_code}: {resp.text}")

    def test_auto_derive_missing_loan_percent_income(self):
        """If loan_percent_income is absent from CSV, service derives it from loan_amnt and person_income."""
        df = self._generate_valid_df(5)
        df_no_lpi = df.drop(columns=["loan_percent_income"])
        csv_bytes = self._df_to_csv_bytes(df_no_lpi)

        resp = self.client.post(
            "/batch-predict",
            files={"file": ("no_lpi.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200, f"Auto-derive failed: {resp.text}")
        validated = BatchPredictionResponse.model_validate(resp.json())
        self.assertEqual(validated.total_records, 5)

    def test_corrupted_file_extensions_and_empty_filename(self):
        """Non-CSV formats (.json, .bin, empty filename) return client error (400/422)."""
        resp_json = self.client.post(
            "/batch-predict",
            files={"file": ("test.json", io.BytesIO(b'{"key": 1}'), "application/json")},
        )
        self.assertEqual(resp_json.status_code, 400)

        resp_no_name = self.client.post(
            "/batch-predict",
            files={"file": ("", io.BytesIO(b"data"), "text/csv")},
        )
        self.assertIn(resp_no_name.status_code, [400, 422])

if __name__ == "__main__":
    unittest.main()

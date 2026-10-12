import itertools
import unittest
import warnings

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except ImportError:
    pass

from fastapi.testclient import TestClient
from app.api.main import app
from app.schemas.credit import PredictionResponse


class TestMilestone2AdversarialStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_all_16_boundary_combinations(self):
        """
        Stress-test all 2^4 = 16 boundary combinations of:
        - person_age in [18, 100]
        - person_income in [0.0, 2000000.0]
        - loan_amnt in [500.0, 50000.0]
        - cb_person_cred_hist_length in [0, 50]

        Verify HTTP 200, clean serialization (0 PydanticSerializationError),
        and strict response schema conformity.
        """
        ages = [18, 100]
        incomes = [0.0, 2000000.0]
        loan_amnts = [500.0, 50000.0]
        hist_lengths = [0, 50]

        base_template = {
            "person_home_ownership": "RENT",
            "person_emp_length": 5.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_int_rate": 11.5,
            "loan_percent_income": None,
            "cb_person_default_on_file": "N",
        }

        tested_count = 0
        for age, income, loan_amnt, hist_len in itertools.product(ages, incomes, loan_amnts, hist_lengths):
            payload = dict(base_template)
            payload["person_age"] = age
            payload["person_income"] = income
            payload["loan_amnt"] = loan_amnt
            payload["cb_person_cred_hist_length"] = hist_len

            resp = self.client.post("/predict", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                f"Boundary failed HTTP 200 for payload: age={age}, income={income}, amnt={loan_amnt}, hist={hist_len}. Status={resp.status_code}, Body={resp.text}"
            )

            data = resp.json()
            # Validate against strict Pydantic model
            validated = PredictionResponse.model_validate(data)
            self.assertIsNotNone(validated)

            # Check predictions map
            self.assertIn("xgboost", data["predictions"])
            self.assertIn("logistic_regression", data["predictions"])
            self.assertIn("decision_tree", data["predictions"])
            self.assertIn("random_forest", data["predictions"])

            # Check probability bounds
            prob = data["final_model"]["probability"]
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertIn(data["final_model"]["decision"], ["Default", "Repaid"])
            self.assertIn(data["final_model"]["risk_band"], ["Low", "Moderate", "High", "Critical"])

            # Check top_factors serialization: no numpy types, all native Python types
            self.assertIsInstance(data["top_factors"], list)
            for factor in data["top_factors"]:
                self.assertIsInstance(factor["feature"], str)
                self.assertIsInstance(factor["contribution"], float)
                self.assertIn(factor["direction"], ["increases_risk", "decreases_risk"])
                # value can be int, float, str
                self.assertTrue(isinstance(factor["value"], (int, float, str, type(None))))

            # Check applicant_data serialization: all native Python types
            for k, v in data["applicant_data"].items():
                self.assertTrue(
                    isinstance(v, (int, float, str, bool, type(None))),
                    f"applicant_data key {k} has unexpected type: {type(v)}: {v}"
                )

            tested_count += 1

        self.assertEqual(tested_count, 16)

    def test_extended_boundary_combinations(self):
        """
        Stress-test additional extreme boundaries:
        - person_emp_length: 0.0 and 60.0
        - loan_int_rate: 1.0 and 40.0
        - loan_percent_income: 0.0 and 1.5
        - all home ownership and loan intents and grades
        """
        extended_cases = [
            # Zero income with explicit loan_percent_income 1.5
            {
                "person_age": 18,
                "person_income": 0.0,
                "person_home_ownership": "OWN",
                "person_emp_length": 0.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "A",
                "loan_amnt": 500.0,
                "loan_int_rate": 1.0,
                "loan_percent_income": 1.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 0,
            },
            # Maximum age, maximum income, maximum loan, maximum cred hist, max emp length, max rate
            {
                "person_age": 100,
                "person_income": 2000000.0,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 60.0,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "G",
                "loan_amnt": 50000.0,
                "loan_int_rate": 40.0,
                "loan_percent_income": 0.025,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 50,
            },
            # Minimal loan amount with maximum percent income 1.5
            {
                "person_age": 50,
                "person_income": 1000.0,
                "person_home_ownership": "OTHER",
                "person_emp_length": 25.0,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "F",
                "loan_amnt": 1500.0,
                "loan_int_rate": 25.0,
                "loan_percent_income": 1.5,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 25,
            },
            # Zero percent income
            {
                "person_age": 30,
                "person_income": 1000000.0,
                "person_home_ownership": "OWN",
                "person_emp_length": 10.0,
                "loan_intent": "VENTURE",
                "loan_grade": "C",
                "loan_amnt": 500.0,
                "loan_int_rate": 5.0,
                "loan_percent_income": 0.0,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 10,
            },
        ]

        for idx, payload in enumerate(extended_cases):
            resp = self.client.post("/predict", json=payload)
            self.assertEqual(resp.status_code, 200, f"Extended case {idx} failed: {resp.text}")
            data = resp.json()
            PredictionResponse.model_validate(data)

    def test_invalid_inputs_return_422_with_validation_details(self):
        """
        Stress-test invalid inputs on POST /predict.
        Verify that all invalid inputs return HTTP 422 with structured validation details
        rather than 500 crashes.
        """
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

        invalid_cases = [
            # person_age out of range / invalid types
            ({"person_age": 17}, "person_age under 18"),
            ({"person_age": 101}, "person_age over 100"),
            ({"person_age": -5}, "negative person_age"),
            ({"person_age": "twenty"}, "string person_age"),
            # person_income out of range / invalid types
            ({"person_income": -0.01}, "negative person_income"),
            ({"person_income": -5000.0}, "large negative person_income"),
            ({"person_income": "rich"}, "string person_income"),
            # person_home_ownership invalid literals
            ({"person_home_ownership": "MANSION"}, "invalid home ownership"),
            ({"person_home_ownership": ""}, "empty home ownership"),
            ({"person_home_ownership": 123}, "numeric home ownership"),
            # person_emp_length out of range
            ({"person_emp_length": -0.5}, "negative emp_length"),
            ({"person_emp_length": 60.1}, "emp_length over 60"),
            ({"person_emp_length": "ten"}, "string emp_length"),
            # loan_intent invalid literals
            ({"loan_intent": "VACATION"}, "invalid loan_intent"),
            ({"loan_intent": "GAMBLING"}, "invalid loan_intent"),
            ({"loan_intent": 99}, "numeric loan_intent"),
            # loan_grade invalid literals
            ({"loan_grade": "H"}, "grade H"),
            ({"loan_grade": "Z"}, "grade Z"),
            ({"loan_grade": "A+"}, "grade A+"),
            # loan_amnt out of range
            ({"loan_amnt": 499.99}, "loan_amnt under 500"),
            ({"loan_amnt": 0.0}, "zero loan_amnt"),
            ({"loan_amnt": -500.0}, "negative loan_amnt"),
            ({"loan_amnt": 50000.01}, "loan_amnt over 50000"),
            ({"loan_amnt": 1000000.0}, "loan_amnt huge"),
            ({"loan_amnt": "ten_k"}, "string loan_amnt"),
            # loan_int_rate out of range
            ({"loan_int_rate": 0.99}, "int_rate under 1.0"),
            ({"loan_int_rate": 0.0}, "int_rate zero"),
            ({"loan_int_rate": -1.0}, "negative int_rate"),
            ({"loan_int_rate": 40.01}, "int_rate over 40.0"),
            ({"loan_int_rate": 100.0}, "huge int_rate"),
            # loan_percent_income out of range
            ({"loan_percent_income": -0.01}, "negative percent income"),
            ({"loan_percent_income": 1.51}, "percent income over 1.5"),
            ({"loan_percent_income": 10.0}, "huge percent income"),
            # cb_person_default_on_file invalid literals
            ({"cb_person_default_on_file": "YES"}, "default on file YES"),
            ({"cb_person_default_on_file": "NO"}, "default on file NO"),
            ({"cb_person_default_on_file": "TRUE"}, "default on file TRUE"),
            ({"cb_person_default_on_file": ""}, "empty default on file"),
            # cb_person_cred_hist_length out of range
            ({"cb_person_cred_hist_length": -1}, "negative cred_hist"),
            ({"cb_person_cred_hist_length": 51}, "cred_hist over 50"),
            ({"cb_person_cred_hist_length": 100}, "cred_hist 100"),
            ({"cb_person_cred_hist_length": "five"}, "string cred_hist"),
        ]

        for override, description in invalid_cases:
            payload = base_payload.copy()
            payload.update(override)
            resp = self.client.post("/predict", json=payload)
            self.assertEqual(
                resp.status_code, 422,
                f"Expected 422 for '{description}', got {resp.status_code}. Body: {resp.text}"
            )
            data = resp.json()
            self.assertIn(
                "detail", data,
                f"Response for '{description}' must contain structured 'detail'. Got: {data}"
            )
            self.assertIsInstance(
                data["detail"], list,
                f"Detail for '{description}' should be a validation error list. Got: {data['detail']}"
            )

    def test_malformed_payload_bodies_return_422(self):
        """
        Test malformed payloads such as arrays, non-dict JSON, and invalid types.
        """
        malformed_payloads = [
            ["not", "a", "dict"],
            "raw string instead of json object",
            12345,
        ]

        for payload in malformed_payloads:
            resp = self.client.post("/predict", json=payload)
            self.assertEqual(
                resp.status_code, 422,
                f"Expected 422 for malformed body {payload}, got {resp.status_code}: {resp.text}"
            )
            data = resp.json()
            self.assertIn("detail", data)


if __name__ == "__main__":
    unittest.main()

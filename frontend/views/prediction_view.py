import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from frontend.styles import get_risk_badge

PRESETS = {
    "Prime Low-Risk Borrower": {
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
    },
    "Borderline Moderate-Risk Borrower": {
        "person_age": 27,
        "person_income": 45000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "C",
        "loan_amnt": 12000.0,
        "loan_int_rate": 13.25,
        "loan_percent_income": 0.27,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    },
    "High-Risk Solvency Distressed": {
        "person_age": 23,
        "person_income": 26000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 1.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "E",
        "loan_amnt": 14000.0,
        "loan_int_rate": 18.75,
        "loan_percent_income": 0.54,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }
}

def render_prediction_view(client):
    st.markdown('<div class="gradient-title">Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Evaluate applicant solvency, compare 4 trained model predictions, '
        'and inspect the official Calibrated XGBoost decision.</div>',
        unsafe_allow_html=True
    )

    # Preset selector bar
    col_preset, col_reset = st.columns([4, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Load Sample Profile Preset:",
            options=["Custom Profile"] + list(PRESETS.keys()),
            index=0
        )
    with col_reset:
        st.write("")
        st.write("")
        if st.button("Reset Form", use_container_width=True):
            st.session_state["applicant_form"] = PRESETS["Prime Low-Risk Borrower"].copy()
            st.rerun()

    # Load preset values if selected
    if preset_choice in PRESETS:
        form_defaults = PRESETS[preset_choice]
    else:
        form_defaults = st.session_state.get("applicant_form", PRESETS["Prime Low-Risk Borrower"])

    with st.form("applicant_prediction_form"):
        st.markdown("### Applicant & Loan Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Information**")
            age = st.number_input(
                "Age (years):", min_value=18, max_value=100,
                value=int(form_defaults["person_age"])
            )
            income = st.number_input(
                "Annual Income ($):", min_value=1000.0, max_value=2000000.0,
                step=2500.0, value=float(form_defaults["person_income"])
            )
            home_ownership = st.selectbox(
                "Home Ownership Status:",
                options=["RENT", "MORTGAGE", "OWN", "OTHER"],
                index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(form_defaults["person_home_ownership"])
            )
            emp_length = st.number_input(
                "Employment Tenure (years):", min_value=0.0, max_value=60.0,
                step=0.5, value=float(form_defaults["person_emp_length"])
            )

        with col2:
            st.markdown("**Loan Terms**")
            loan_amnt = st.number_input(
                "Requested Loan Amount ($):", min_value=500.0, max_value=50000.0,
                step=500.0, value=float(form_defaults["loan_amnt"])
            )
            loan_intent = st.selectbox(
                "Loan Intent:",
                options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                index=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(form_defaults["loan_intent"])
            )
            loan_grade = st.selectbox(
                "Loan Risk Grade:",
                options=["A", "B", "C", "D", "E", "F", "G"],
                index=["A", "B", "C", "D", "E", "F", "G"].index(form_defaults["loan_grade"])
            )
            int_rate = st.number_input(
                "Interest Rate (%):", min_value=1.0, max_value=40.0,
                step=0.25, value=float(form_defaults["loan_int_rate"])
            )

        with col3:
            st.markdown("**Credit History & Solvency**")
            calculated_ratio = round(loan_amnt / max(income, 1.0), 2)
            ratio_display = st.number_input(
                f"Debt-to-Income Ratio (computed: {calculated_ratio:.2f}):",
                min_value=0.0, max_value=1.5, step=0.01,
                value=float(form_defaults.get("loan_percent_income", calculated_ratio))
            )
            default_on_file = st.selectbox(
                "Historical Default on File:",
                options=["N", "Y"],
                index=0 if form_defaults["cb_person_default_on_file"] == "N" else 1
            )
            cred_hist_len = st.number_input(
                "Credit History Length (years):", min_value=0, max_value=50,
                value=int(form_defaults["cb_person_cred_hist_length"])
            )
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submit_btn or "last_prediction" in st.session_state:
        if submit_btn:
            applicant_payload = {
                "person_age": int(age),
                "person_income": float(income),
                "person_home_ownership": home_ownership,
                "person_emp_length": float(emp_length),
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": float(loan_amnt),
                "loan_int_rate": float(int_rate),
                "loan_percent_income": float(ratio_display),
                "cb_person_default_on_file": default_on_file,
                "cb_person_cred_hist_length": int(cred_hist_len)
            }
            with st.spinner("Executing 4-model inference and calculating local SHAP attributions..."):
                pred_result = client.predict(applicant_payload)
                st.session_state["last_prediction"] = pred_result
                st.session_state["applicant_form"] = applicant_payload.copy()
        else:
            pred_result = st.session_state["last_prediction"]

        # Display Final Model Result in Champion Glass Card
        final = pred_result["final_model"]
        prob = final["probability"]
        decision = final["decision"]
        band = final["risk_band"]
        thresh = final["threshold"]

        st.markdown("---")
        st.markdown("### Official Decision Support System Output")
        
        col_res1, col_res2, col_res3 = st.columns([2, 1.5, 1.5])
        with col_res1:
            st.markdown(
                f"""
                <div class="glass-highlight">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span class="badge badge-champion">Champion Model: {final['name']}</span>
                        {get_risk_badge(band)}
                    </div>
                    <div style="font-size:2.6rem; font-weight:800; color:#f8fafc; margin-bottom:6px;">
                        {prob*100:.1f}% <span style="font-size:1.1rem; color:#94a3b8; font-weight:500;">Default Probability</span>
                    </div>
                    <div style="color:#cbd5e1; font-size:1.0rem;">
                        <strong>Recommendation:</strong> {('DENY / HIGH RISK OF DEFAULT' if decision == 'Default' else 'APPROVE / FAVORABLE REPAYMENT PROFILE')}
                    </div>
                    <div style="color:#64748b; font-size:0.85rem; margin-top:8px;">
                        Optimized Operating Threshold: <strong>{thresh:.4f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_res2:
            st.markdown(
                f"""
                <div class="glass-container" style="height:92%;">
                    <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase; margin-bottom:8px;">Model Certainty</div>
                    <div style="font-size:1.7rem; font-weight:700; color:#e2e8f0; margin-bottom:10px;">
                        {('High Risk' if prob >= thresh else 'Safe Tier')}
                    </div>
                    <div style="color:#94a3b8; font-size:0.9rem;">
                        Probability buffer to decision boundary: <strong>{abs(prob - thresh)*100:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_res3:
            st.markdown(
                f"""
                <div class="glass-container" style="height:92%;">
                    <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase; margin-bottom:8px;">Calibration Status</div>
                    <div style="font-size:1.7rem; font-weight:700; color:#38bdf8; margin-bottom:10px;">
                        Sigmoid Calibrated
                    </div>
                    <div style="color:#94a3b8; font-size:0.9rem;">
                        Empirical Brier score: <strong>0.0661</strong> (well-calibrated)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 4-Model Side-by-Side Comparison
        st.markdown("### Four-Model Side-by-Side Comparison")
        st.markdown("Observation of all 4 trained model architectures on this identical applicant record:")

        preds = pred_result["predictions"]
        comp_cols = st.columns(4)

        models_display = [
            ("Logistic Regression", preds.get("logistic_regression", 0.0), "Linear Baseline", False),
            ("Decision Tree", preds.get("decision_tree", 0.0), "Nonlinear Tree", False),
            ("Random Forest", preds.get("random_forest", 0.0), "Bagged Ensemble", False),
            ("XGBoost (Calibrated)", preds.get("xgboost", 0.0), "Official Final Model", True),
        ]

        for i, (mname, mprob, desc, is_champion) in enumerate(models_display):
            with comp_cols[i]:
                card_class = "glass-highlight" if is_champion else "glass-container"
                badge_html = '<span class="badge badge-champion">FINAL</span>' if is_champion else '<span style="color:#64748b; font-size:0.8rem;">BASELINE</span>'
                st.markdown(
                    f"""
                    <div class="{card_class}" style="padding:16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="font-size:0.95rem; color:#f1f5f9;">{mname}</strong>
                            {badge_html}
                        </div>
                        <div style="color:#94a3b8; font-size:0.75rem; margin-top:2px;">{desc}</div>
                        <div style="font-size:2.0rem; font-weight:800; color:{('#60a5fa' if is_champion else '#e2e8f0')}; margin:12px 0;">
                            {mprob*100:.1f}%
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8;">
                            Predicted: <strong>{('Default' if mprob >= thresh else 'Repaid')}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Comparative Bar Chart
        comp_df = pd.DataFrame([
            {"Model": "Logistic Regression", "Default Probability (%)": preds.get("logistic_regression", 0.0)*100, "Type": "Baseline"},
            {"Model": "Decision Tree", "Default Probability (%)": preds.get("decision_tree", 0.0)*100, "Type": "Baseline"},
            {"Model": "Random Forest", "Default Probability (%)": preds.get("random_forest", 0.0)*100, "Type": "Baseline"},
            {"Model": "XGBoost (Calibrated)", "Default Probability (%)": preds.get("xgboost", 0.0)*100, "Type": "Final Selected"}
        ])

        chart = alt.Chart(comp_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0, title="Model Architecture")),
            y=alt.Y("Default Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=["Baseline", "Final Selected"], range=["#475569", "#3b82f6"]),
                legend=alt.Legend(title="Model Tier")
            ),
            tooltip=["Model", "Default Probability (%)", "Type"]
        ).properties(height=260)

        # Add Threshold Rule Line
        rule = alt.Chart(pd.DataFrame([{"y": thresh * 100}])).mark_rule(
            color="#ef4444", strokeDash=[6, 4], strokeWidth=2
        ).encode(y="y:Q")

        st.altair_chart(chart + rule, use_container_width=True)

        # Local Explainability Section
        st.markdown("### Local Explainability (SHAP Risk Factors)")
        st.markdown(
            "Feature contributions explaining **why** the calibrated XGBoost model produced this probability estimate:"
        )

        factors = pred_result.get("top_factors", [])
        if factors:
            f_col1, f_col2 = st.columns(2)
            
            risk_increasing = [f for f in factors if f["direction"] == "increases_risk"]
            risk_decreasing = [f for f in factors if f["direction"] == "decreases_risk"]

            with f_col1:
                st.markdown("##### 🔺 Top Risk-Increasing Factors")
                if risk_increasing:
                    for f in risk_increasing:
                        st.markdown(
                            f"""
                            <div class="glass-container" style="padding:14px; margin-bottom:10px; border-left:4px solid #ef4444;">
                                <div style="display:flex; justify-content:space-between;">
                                    <strong>{f['feature_name']}</strong>
                                    <span style="color:#ef4444; font-weight:700;">+{abs(f['contribution']):.3f}</span>
                                </div>
                                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">{f['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No major risk-elevating drivers detected for this profile.")

            with f_col2:
                st.markdown("##### 🔻 Top Risk-Decreasing Factors")
                if risk_decreasing:
                    for f in risk_decreasing:
                        st.markdown(
                            f"""
                            <div class="glass-container" style="padding:14px; margin-bottom:10px; border-left:4px solid #10b981;">
                                <div style="display:flex; justify-content:space-between;">
                                    <strong>{f['feature_name']}</strong>
                                    <span style="color:#10b981; font-weight:700;">-{abs(f['contribution']):.3f}</span>
                                </div>
                                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">{f['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No significant risk-mitigating factors present.")
    else:
        st.markdown("---")
        st.markdown(
            """
            <div class="glass-container" style="text-align: center; padding: 42px 24px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🎯</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px;">Awaiting Applicant Evaluation</h4>
                <p style="color: #94a3b8; max-width: 580px; margin: 0 auto 20px auto; font-size: 0.95rem; line-height: 1.5;">
                    Select one of the sample borrower presets above or customize the applicant characteristics,
                    then click <strong>Generate Prediction</strong> to run 4-model inference, observe side-by-side risk scores,
                    and inspect local SHAP factor attributions.
                </p>
                <div style="display: inline-flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
                    <span class="badge" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3);">
                        ⚡ 4-Model Inference
                    </span>
                    <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">
                        ⚖️ Sigmoid Calibrated Probabilities
                    </span>
                    <span class="badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3);">
                        🔍 Local TreeSHAP Explanations
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

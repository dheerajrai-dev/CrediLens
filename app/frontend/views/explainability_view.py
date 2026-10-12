import streamlit as st
import pandas as pd
import altair as alt

def render_explainability_view(client):
    st.markdown('<div class="gradient-title">Model Explainability & SHAP Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Understand both global feature drivers and local applicant decision paths '
        'powered by TreeSHAP attribution analysis.</div>',
        unsafe_allow_html=True
    )

    expl_data = client.get_explainability()
    global_items = expl_data.get("global_importance", [])
    methodology = expl_data.get("methodology", "")
    disclaimer = expl_data.get("disclaimer", "")

    # Top Overview Cards
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("### Global Feature Impact (TreeSHAP)")
        st.markdown(
            "Ranking of the most influential risk signals utilized by the trained XGBoost model across the dataset. "
            "Values represent the mean absolute SHAP contribution |SHAP| in log-odds scale."
        )

        df_global = pd.DataFrame(global_items)
        chart = alt.Chart(df_global).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
            y=alt.Y("label:N", sort="-x", title="Credit Signal"),
            x=alt.X("importance:Q", title="Relative Impact Weight"),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["label", "importance", "trend", "business_impact"]
        ).properties(height=340)

        st.altair_chart(chart, width="stretch")

    with col2:
        st.markdown("### Signal Interpretation Guide")
        st.markdown(
            """
            <div class="glass-container">
                <h5 style="color:#60a5fa; margin-bottom:12px;">Top Risk Drivers</h5>
                <p style="font-size:0.88rem; color:#cbd5e1; margin-bottom:12px;">
                    <strong>1. Loan-to-Income Ratio:</strong> The single strongest predictor. High debt service ratios severely constrain cash flows and multiply default risk.
                </p>
                <p style="font-size:0.88rem; color:#cbd5e1; margin-bottom:12px;">
                    <strong>2. Interest Rate:</strong> Higher rates reflect baseline lender risk pricing and escalate monthly repayment hurdle.
                </p>
                <p style="font-size:0.88rem; color:#cbd5e1; margin-bottom:12px;">
                    <strong>3. Annual Income:</strong> Substantial income acts as a cushion absorbing macro-economic or employment volatility.
                </p>
                <p style="font-size:0.88rem; color:#cbd5e1;">
                    <strong>4. Credit Grade & History:</strong> Historical bureau flags (prior default) impose substantial non-linear penalties.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detailed Global Feature Table
    st.markdown("### Comprehensive Feature Attribution Directory")
    st.dataframe(
        df_global[["label", "importance", "trend", "business_impact"]].rename(columns={
            "label": "Feature Name",
            "importance": "Relative Weight",
            "trend": "Directional Effect",
            "business_impact": "Credit Underwriting Impact"
        }),
        width="stretch",
        hide_index=True
    )

    # Interactive Local Simulator
    st.markdown("### Interactive Local Attribution Simulator")
    st.markdown(
        "Explore how altering key applicant characteristics dynamically shifts the model's log-odds output "
        "and SHAP attribution breakdown."
    )

    with st.expander("🔬 Open Interactive What-If Scenario Simulator", expanded=True):
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_income = st.slider("Annual Income ($):", min_value=15000, max_value=180000, value=50000, step=5000)
            sim_loan = st.slider("Loan Amount ($):", min_value=1000, max_value=40000, value=12000, step=1000)
        with sim_c2:
            sim_rate = st.slider("Interest Rate (%):", min_value=5.0, max_value=25.0, value=12.5, step=0.5)
            sim_emp = st.slider("Employment Tenure (yrs):", min_value=0, max_value=25, value=3, step=1)
        with sim_c3:
            sim_home = st.selectbox("Home Ownership:", ["RENT", "MORTGAGE", "OWN"], index=0)
            sim_def = st.selectbox("Historical Default:", ["N", "Y"], index=0)

        # Run quick prediction for simulator
        sim_ratio = round(sim_loan / max(sim_income, 1), 2)
        sim_payload = {
            "person_age": 28,
            "person_income": float(sim_income),
            "person_home_ownership": sim_home,
            "person_emp_length": float(sim_emp),
            "loan_intent": "PERSONAL",
            "loan_grade": "C" if sim_rate > 11 else "B",
            "loan_amnt": float(sim_loan),
            "loan_int_rate": float(sim_rate),
            "loan_percent_income": float(sim_ratio),
            "cb_person_default_on_file": sim_def,
            "cb_person_cred_hist_length": 4
        }
        sim_res = client.predict(sim_payload)
        sim_final = sim_res["final_model"]

        st.markdown(
            f"""
            <div class="glass-highlight" style="margin-top:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.9rem; color:#94a3b8;">Simulated Loan-to-Income: <strong>{sim_ratio*100:.1f}%</strong></span>
                        <div style="font-size:1.8rem; font-weight:700; color:#f8fafc;">
                            Simulated Default Probability: <span style="color:#60a5fa;">{sim_final['probability']*100:.1f}%</span>
                        </div>
                    </div>
                    <div>
                        <span class="badge badge-{sim_final['risk_band'].lower()}">{sim_final['risk_band']} Risk</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        sim_factors = sim_res.get("top_factors", [])
        if sim_factors:
            sf_df = pd.DataFrame(sim_factors)
            sf_df["value"] = sf_df["value"].astype(str)
            sf_chart = alt.Chart(sf_df).mark_bar().encode(
                y=alt.Y("feature_name:N", sort=None, title="Risk Factor"),
                x=alt.X("contribution:Q", title="Log-Odds Shift (+ Raises Risk, - Lowers Risk)"),
                color=alt.Color("direction:N", scale=alt.Scale(
                    domain=["increases_risk", "decreases_risk"],
                    range=["#ef4444", "#10b981"]
                ), legend=alt.Legend(title="Attribution Direction")),
                tooltip=["feature_name", "value", "contribution", "description"]
            ).properties(height=220)

            st.altair_chart(sf_chart, width="stretch")

    # Attribution Disclaimer Card
    st.markdown("---")
    st.markdown(
        f"""
        <div class="glass-container" style="border-left:4px solid #6366f1;">
            <strong style="color:#818cf8; font-size:0.95rem;">Methodology & Attribution Notice</strong>
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:6px; line-height:1.5;">
                {methodology}<br><br>
                <em>{disclaimer}</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

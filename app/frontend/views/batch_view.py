import streamlit as st
import pandas as pd
import io
import altair as alt

from app.frontend.styles import get_risk_badge

SAMPLE_CSV_DATA = """person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length
22,59000,RENT,12.0,PERSONAL,D,35000,16.02,0.59,Y,3
21,9600,OWN,5.0,EDUCATION,B,1000,11.14,0.10,N,2
25,9600,MORTGAGE,1.0,MEDICAL,C,5500,12.87,0.57,N,3
23,65500,RENT,4.0,MEDICAL,C,35000,15.23,0.53,N,2
24,54400,RENT,8.0,MEDICAL,C,35000,14.27,0.55,Y,4
32,75000,MORTGAGE,6.0,VENTURE,A,12000,8.49,0.16,N,7
28,48000,RENT,3.0,EDUCATION,B,8000,10.25,0.17,N,4
45,115000,OWN,14.0,HOMEIMPROVEMENT,A,20000,7.15,0.17,N,12
30,32000,RENT,2.0,DEBTCONSOLIDATION,D,15000,17.49,0.47,Y,3
39,92000,MORTGAGE,9.0,PERSONAL,B,18000,9.99,0.20,N,9
"""

def render_batch_view(client):
    st.markdown('<div class="gradient-title">Batch Prediction Utility</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Batch score multi-applicant CSV portfolios using exclusively '
        'the final calibrated XGBoost model, with automated risk band stratification.</div>',
        unsafe_allow_html=True
    )

    # Info card emphasizing the architecture rule: ONLY the final model runs
    st.markdown(
        """
        <div class="glass-container" style="border-left:4px solid #3b82f6;">
            <div style="font-size:0.9rem; color:#cbd5e1;">
                <strong>Architectural Note:</strong> As per the system specification, batch scoring is powered
                <strong>strictly by the final selected model</strong> (Calibrated XGBoost) to maintain high-throughput
                production efficiency. Multi-model evaluation belongs to the Model Evaluation view.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_up, col_tpl = st.columns([3, 1.5])
    with col_tpl:
        st.markdown("##### Need Test Data?")
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=SAMPLE_CSV_DATA,
            file_name="sample_applicants.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Contains 10 diverse applicant rows ready for direct batch testing.")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Applicant CSV File:",
            type=["csv", "txt"],
            help="File must contain the 11 documented applicant and loan feature columns."
        )

    # Demo quick-test button
    if st.button("⚡ Quick Test with 10 Sample Applicants"):
        uploaded_file = io.StringIO(SAMPLE_CSV_DATA)

    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, io.StringIO):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_csv(uploaded_file)

            st.markdown(f"**Loaded Preview ({len(df_input)} rows):**")
            st.dataframe(df_input.head(5), use_container_width=True)

            if len(df_input) == 0:
                st.warning("Uploaded CSV contains no data rows to score.")
                return

            with st.spinner("Scoring portfolio with Calibrated XGBoost..."):
                batch_response = client.predict_batch(df_input)

            total = batch_response["total_records"]
            defaults = batch_response["predicted_defaults"]
            repays = batch_response["predicted_repays"]
            def_rate = batch_response["default_rate"]
            avg_risk = batch_response["average_risk"]
            results = batch_response["results"]

            if len(results) == 0:
                st.warning("Uploaded CSV contains no data rows to score.")
                return

            st.markdown("---")
            st.markdown("### Portfolio Scoring Summary")

            # Metric pills
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-pill">
                        <div class="label">Total Evaluated</div>
                        <div class="value">{total:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-pill">
                        <div class="label">Predicted Defaults</div>
                        <div class="value" style="color:#f87171;">{defaults:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-pill">
                        <div class="label">Approved / Repaid</div>
                        <div class="value" style="color:#34d399;">{repays:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="metric-pill">
                        <div class="label">Portfolio Default Rate</div>
                        <div class="value" style="color:#60a5fa;">{def_rate*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Build enriched results DataFrame
            res_df = pd.DataFrame(results)
            # Combine original input columns with scoring output
            combined_df = pd.concat([
                res_df[["applicant_id", "default_probability", "decision", "risk_band"]].rename(columns={
                    "applicant_id": "Applicant_ID",
                    "default_probability": "Default_Probability",
                    "decision": "Decision",
                    "risk_band": "Risk_Band"
                }),
                df_input.reset_index(drop=True)
            ], axis=1)

            st.markdown("### Detailed Scoring Results")
            st.dataframe(
                combined_df,
                use_container_width=True,
                hide_index=True
            )

            # Risk Band Distribution Chart
            st.markdown("### Portfolio Risk Tier Distribution")
            band_counts = res_df["risk_band"].value_counts().reset_index()
            band_counts.columns = ["Risk_Band", "Count"]

            band_chart = alt.Chart(band_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("Risk_Band:N", sort=["Low", "Moderate", "High", "Critical"], title="Risk Tier"),
                y=alt.Y("Count:Q", title="Applicant Count"),
                color=alt.Color("Risk_Band:N", scale=alt.Scale(
                    domain=["Low", "Moderate", "High", "Critical"],
                    range=["#10b981", "#f59e0b", "#f97316", "#ef4444"]
                ), legend=None),
                tooltip=["Risk_Band", "Count"]
            ).properties(height=240)

            st.altair_chart(band_chart, use_container_width=True)

            # Download Result CSV Button
            out_csv = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Portfolio Results (CSV)",
                data=out_csv,
                file_name="credilens_scored_batch.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
    else:
        st.markdown("---")
        st.markdown(
            """
            <div class="glass-container" style="text-align: center; padding: 42px 24px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📁</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px;">Upload an Applicant Portfolio to Score</h4>
                <p style="color: #94a3b8; max-width: 620px; margin: 0 auto 20px auto; font-size: 0.95rem; line-height: 1.5;">
                    Upload a CSV file containing applicant and loan attributes to evaluate credit risk in bulk.
                    Each row will be scored through the calibrated XGBoost model, assigned an empirical default probability,
                    and categorized into an underwriting risk tier.
                </p>
                <div style="background: rgba(15, 23, 42, 0.4); border-radius: 8px; padding: 14px; max-width: 650px; margin: 0 auto 16px auto; text-align: left; font-size: 0.83rem; color: #94a3b8;">
                    <strong style="color: #60a5fa;">Required CSV Columns:</strong><br>
                    <code>person_age</code>, <code>person_income</code>, <code>person_home_ownership</code>, <code>person_emp_length</code>, <code>loan_intent</code>, <code>loan_grade</code>, <code>loan_amnt</code>, <code>loan_int_rate</code>, <code>loan_percent_income</code>, <code>cb_person_default_on_file</code>, <code>cb_person_cred_hist_length</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

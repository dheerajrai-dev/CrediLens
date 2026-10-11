import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def render_evaluation_view(client):
    st.markdown('<div class="gradient-title">Model Evaluation & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Comprehensive experimental evaluation of all 4 model families on '
        'the credit default dataset, documenting model selection criteria and calibration verification.</div>',
        unsafe_allow_html=True
    )

    eval_data = client.get_evaluation()
    table = eval_data.get("comparison_table", [])
    cms = eval_data.get("confusion_matrices", {})
    curves = eval_data.get("curves", {})
    selected_model = eval_data.get("selected_model", "XGBoost (Tuned + Calibrated)")
    rationale = eval_data.get("selection_rationale", "")

    # Top summary metrics table
    st.markdown("### Four-Model Benchmark Metrics")
    df_metrics = pd.DataFrame(table)

    # Format dataframe for display
    display_df = df_metrics.copy()
    display_df.columns = [
        "Model Architecture", "ROC-AUC", "PR-AUC", "Precision",
        "Recall", "F1-Score", "Accuracy", "Brier Score"
    ]

    st.dataframe(
        display_df.style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score", "Accuracy"],
            color="rgba(37, 99, 235, 0.3)"
        ).highlight_min(
            subset=["Brier Score"],
            color="rgba(16, 185, 129, 0.3)"
        ),
        use_container_width=True,
        hide_index=True
    )

    # Multi-metric comparative bar chart
    st.markdown("### Metric Comparison Across Architectures")
    
    # Melt dataframe for Altair chart
    chart_data = df_metrics.melt(
        id_vars=["model"],
        value_vars=["roc_auc", "pr_auc", "precision", "recall", "f1_score"],
        var_name="Metric",
        value_name="Score"
    )
    metric_map = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score"
    }
    chart_data["Metric"] = chart_data["Metric"].map(metric_map)

    grouped_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-15)),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.4, 1.0])),
        color=alt.Color("model:N", legend=alt.Legend(title="Model Family")),
        column=alt.Column("Metric:N", title="Evaluation Metric")
    ).properties(width=140, height=260)

    st.altair_chart(grouped_chart)

    # Model Selection Rationale Banner
    st.markdown("### Model Selection Rationale")
    st.markdown(
        f"""
        <div class="glass-highlight">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                <span class="badge badge-champion">Selected Final Model</span>
                <strong style="font-size:1.15rem; color:#f8fafc;">{selected_model}</strong>
            </div>
            <div style="color:#cbd5e1; font-size:0.95rem; line-height:1.6;">
                {rationale}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Interactive Curves Tabs
    st.markdown("### Diagnostic Curves & Calibration")
    curve_tab1, curve_tab2, curve_tab3 = st.tabs([
        "📈 ROC Curves (All 4 Models)",
        "🎯 Precision-Recall Curves",
        "⚖️ Calibration & Brier Score"
    ])

    with curve_tab1:
        st.markdown(
            "Receiver Operating Characteristic (ROC) curves plotting True Positive Rate vs False Positive Rate. "
            "Higher area under the curve (AUC) indicates superior class separability."
        )
        roc_pts = curves.get("roc", {})
        roc_rows = []
        for mkey, mname in [
            ("logistic_regression", "Logistic Regression (0.8705)"),
            ("decision_tree", "Decision Tree (0.8417)"),
            ("random_forest", "Random Forest (0.9284)"),
            ("xgboost", "XGBoost Calibrated (0.9471)")
        ]:
            if mkey in roc_pts:
                fprs = roc_pts[mkey]["fpr"]
                tprs = roc_pts[mkey]["tpr"]
                for f, t in zip(fprs, tprs):
                    roc_rows.append({"FPR": f, "TPR": t, "Model": mname})

        roc_df = pd.DataFrame(roc_rows)
        roc_chart = alt.Chart(roc_df).mark_line(strokeWidth=3).encode(
            x=alt.X("FPR:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="True Positive Rate (Recall)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(
                domain=[
                    "Logistic Regression (0.8705)",
                    "Decision Tree (0.8417)",
                    "Random Forest (0.9284)",
                    "XGBoost Calibrated (0.9471)"
                ],
                range=["#94a3b8", "#f59e0b", "#10b981", "#3b82f6"]
            )),
            tooltip=["Model", "FPR", "TPR"]
        ).properties(height=360)

        # Diagonal baseline
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#64748b"
        ).encode(x="x:Q", y="y:Q")

        st.altair_chart(roc_chart + diag, use_container_width=True)

    with curve_tab2:
        st.markdown(
            "Precision-Recall (PR) curves evaluating performance under class imbalance (~21.8% defaults). "
            "XGBoost maintains exceptional precision across operational recall ranges, minimizing costly false defaults."
        )
        pr_pts = curves.get("pr", {})
        pr_rows = []
        for mkey, mname in [
            ("logistic_regression", "Logistic Regression (0.7135)"),
            ("decision_tree", "Decision Tree (0.6052)"),
            ("random_forest", "Random Forest (0.8803)"),
            ("xgboost", "XGBoost Calibrated (0.9043)")
        ]:
            if mkey in pr_pts:
                recalls = pr_pts[mkey]["recall"]
                precisions = pr_pts[mkey]["precision"]
                for r, p in zip(recalls, precisions):
                    pr_rows.append({"Recall": r, "Precision": p, "Model": mname})

        pr_df = pd.DataFrame(pr_rows)
        pr_chart = alt.Chart(pr_df).mark_line(strokeWidth=3).encode(
            x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(
                domain=[
                    "Logistic Regression (0.7135)",
                    "Decision Tree (0.6052)",
                    "Random Forest (0.8803)",
                    "XGBoost Calibrated (0.9043)"
                ],
                range=["#94a3b8", "#f59e0b", "#10b981", "#3b82f6"]
            )),
            tooltip=["Model", "Recall", "Precision"]
        ).properties(height=360)

        st.altair_chart(pr_chart, use_container_width=True)

    with curve_tab3:
        st.markdown(
            "Reliability diagram comparing uncalibrated XGBoost vs Platt/Sigmoid calibrated probabilities. "
            "The calibrated curve aligns closely with the diagonal, reducing the Brier score from 0.089 to 0.0661."
        )
        cal_pts = curves.get("calibration", {})
        cal_rows = []
        for ckey, cname in [
            ("perfect", "Perfect Calibration (Diagonal)"),
            ("uncalibrated", "Uncalibrated XGBoost (Brier: 0.089)"),
            ("calibrated", "Sigmoid Calibrated XGBoost (Brier: 0.0661)")
        ]:
            if ckey in cal_pts:
                preds = cal_pts[ckey]["pred"]
                acts = cal_pts[ckey]["actual"]
                for pr, ac in zip(preds, acts):
                    cal_rows.append({"Mean Predicted Probability": pr, "Fraction of Positives": ac, "Type": cname})

        cal_df = pd.DataFrame(cal_rows)
        cal_chart = alt.Chart(cal_df).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Mean Predicted Probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Fraction of Positives:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=[
                    "Perfect Calibration (Diagonal)",
                    "Uncalibrated XGBoost (Brier: 0.089)",
                    "Sigmoid Calibrated XGBoost (Brier: 0.0661)"
                ],
                range=["#64748b", "#f97316", "#38bdf8"]
            )),
            tooltip=["Type", "Mean Predicted Probability", "Fraction of Positives"]
        ).properties(height=360)

        st.altair_chart(cal_chart, use_container_width=True)

    # Confusion Matrix Explorer (Dropdown switchable as per spec!)
    st.markdown("### Confusion Matrix Explorer")
    st.markdown("Switch between the 4 models to inspect True/False positives and negatives on the 6,305 held-out test applicants:")

    cm_model_choice = st.selectbox(
        "Select Model Architecture for Confusion Matrix:",
        options=list(cms.keys()),
        index=list(cms.keys()).index("XGBoost (Tuned + Calibrated)") if "XGBoost (Tuned + Calibrated)" in cms else 0
    )

    if cm_model_choice in cms:
        cm_data = cms[cm_model_choice]
        col_cm1, col_cm2 = st.columns([1.5, 1])

        with col_cm1:
            st.markdown(
                f"""
                <div class="glass-container">
                    <h5 style="margin-bottom:14px; color:#f8fafc;">Confusion Matrix: {cm_model_choice}</h5>
                    <div class="cm-grid">
                        <div class="cm-tile cm-tn">
                            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">True Negative (TN)</div>
                            <div style="font-size:1.8rem; font-weight:800; color:#34d399;">{cm_data['tn']:,}</div>
                            <div style="font-size:0.75rem; color:#cbd5e1;">Safe loans correctly funded</div>
                        </div>
                        <div class="cm-tile cm-fp">
                            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">False Positive (FP)</div>
                            <div style="font-size:1.8rem; font-weight:800; color:#fbbf24;">{cm_data['fp']:,}</div>
                            <div style="font-size:0.75rem; color:#cbd5e1;">Safe loans rejected (Lost revenue)</div>
                        </div>
                        <div class="cm-tile cm-fn">
                            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">False Negative (FN)</div>
                            <div style="font-size:1.8rem; font-weight:800; color:#f87171;">{cm_data['fn']:,}</div>
                            <div style="font-size:0.75rem; color:#cbd5e1;">Defaults missed (Credit loss)</div>
                        </div>
                        <div class="cm-tile cm-tp">
                            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">True Positive (TP)</div>
                            <div style="font-size:1.8rem; font-weight:800; color:#60a5fa;">{cm_data['tp']:,}</div>
                            <div style="font-size:0.75rem; color:#cbd5e1;">Defaults caught & prevented</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_cm2:
            st.markdown(
                f"""
                <div class="glass-container">
                    <h5 style="margin-bottom:14px; color:#f8fafc;">Operational Performance</h5>
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.85rem;">
                            <span>Classification Accuracy</span>
                            <strong style="color:#f8fafc;">{cm_data['accuracy']*100:.2f}%</strong>
                        </div>
                    </div>
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.85rem;">
                            <span>Precision (Positive Class)</span>
                            <strong style="color:#38bdf8;">{cm_data['precision']*100:.2f}%</strong>
                        </div>
                    </div>
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.85rem;">
                            <span>Recall / Sensitivity</span>
                            <strong style="color:#a78bfa;">{cm_data['recall']*100:.2f}%</strong>
                        </div>
                    </div>
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:0.85rem;">
                            <span>F1-Score</span>
                            <strong style="color:#f472b6;">{cm_data['f1']*100:.2f}%</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

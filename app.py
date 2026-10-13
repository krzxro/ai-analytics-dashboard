"""InsightPilot: interactive analysis of IBM's fictional HR attrition dataset."""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics import clean_data

st.set_page_config(page_title="InsightPilot | Employee Retention", layout="wide", initial_sidebar_state="expanded")
COLORS = {"Stayed": "#20BFA9", "Left": "#F26B74"}
PALETTE = ["#6366F1", "#06B6D4", "#F59E0B", "#EC4899", "#10B981", "#8B5CF6", "#F97316"]
SOURCE = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
DATA = Path(__file__).parent / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1450px;}
h1, h2, h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {border: 1px solid #94a3b83d; border-radius: 12px; padding: 15px;}
</style>""", unsafe_allow_html=True)


def polish(fig, height=410, percent=False):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=12, r=16, t=65, b=20),
                      font=dict(family="Arial", size=13), title_font=dict(size=18),
                      legend_title_text="", hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E9EEF5", zeroline=False)
    if percent:
        fig.update_yaxes(ticksuffix="%")
    return fig


def rate_by(data, feature):
    out = data.groupby(feature, dropna=False).agg(Employees=("AttritionFlag", "size"),
                                                   Departures=("AttritionFlag", "sum"),
                                                   AttritionRate=("AttritionFlag", "mean")).reset_index()
    out["AttritionRate"] = 100 * out["AttritionRate"]
    return out


st.title("InsightPilot | Employee Retention Intelligence")
st.caption("IBM HR Analytics (fictional workforce) · Business intelligence · Predictive modeling · Optional generative AI")
with st.sidebar:
    st.header("Data source")
    st.markdown(f"[Download the IBM HR dataset on Kaggle]({SOURCE})")
    uploaded = st.file_uploader("Upload employee attrition CSV", type="csv")
    st.caption("Upload the original CSV or save it in the project's data folder.")
    st.divider()
    st.header("Dashboard controls")
    st.caption("Filters apply to descriptive dashboards and AI briefing; the predictive model uses the complete dataset.")
    st.divider()
    st.header("AI settings")
    use_ai = st.toggle("Enable AI briefing", value=False)
    model = st.text_input("OpenAI model", value="gpt-4.1-mini")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif DATA.exists():
    df = pd.read_csv(DATA)
else:
    st.info("Upload WA_Fn-UseC_-HR-Employee-Attrition.csv using the sidebar to begin.")
    st.markdown(f"[Get the dataset on Kaggle]({SOURCE})")
    st.stop()

required = {"Attrition", "Department", "JobRole", "OverTime", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany"}
if not required.issubset(df.columns):
    st.error("Dataset is missing required columns: " + ", ".join(sorted(required - set(df.columns))))
    st.stop()

cleaned, audit = clean_data(df)
cleaned["AttritionFlag"] = cleaned["Attrition"].eq("Yes").astype(int)
cleaned["Outcome"] = cleaned["AttritionFlag"].map({0: "Stayed", 1: "Left"})
with st.sidebar:
    departments = sorted(cleaned["Department"].dropna().unique().tolist())
    selected_depts = st.multiselect("Departments", departments, default=departments)
    overtime = st.selectbox("Overtime", ["All", "Yes", "No"])
    age_range = None
    if "Age" in cleaned:
        lo, hi = int(cleaned["Age"].min()), int(cleaned["Age"].max())
        age_range = st.slider("Employee age", lo, hi, (lo, hi))

filtered = cleaned[cleaned["Department"].isin(selected_depts)].copy()
if overtime != "All":
    filtered = filtered[filtered["OverTime"] == overtime]
if age_range is not None:
    filtered = filtered[filtered["Age"].between(*age_range)]
if filtered.empty:
    st.warning("No employees match the current filters. Widen your selections in the sidebar.")
    st.stop()

n = len(filtered)
left = int(filtered["AttritionFlag"].sum())
rate = 100 * left / n
st.caption("Research limitation: IBM-created fictional employee records. All findings are descriptive associations, not causal effects or evidence about actual IBM employees.")

executive, drivers, model_tab, briefing = st.tabs(["Executive overview", "Retention drivers", "Predictive modeling", "AI briefing and exports"])

with executive:
    a, b, c, d = st.columns(4)
    a.metric("Employees in view", f"{n:,}")
    b.metric("Employees who left", f"{left:,}")
    c.metric("Attrition rate", f"{rate:.1f}%")
    d.metric("Median monthly income", f"{filtered['MonthlyIncome'].median():,.0f}")

    c1, c2 = st.columns([1.35, 1])
    with c1:
        dept = rate_by(filtered, "Department").sort_values("AttritionRate")
        fig = px.bar(dept, y="Department", x="AttritionRate", orientation="h", color="Department",
                     color_discrete_sequence=PALETTE, text="AttritionRate", custom_data=["Employees", "Departures"],
                     title="Attrition rate by department")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          hovertemplate="Attrition: %{x:.1f}%<br>Employees: %{customdata[0]}<br>Departures: %{customdata[1]}<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="Attrition rate (%)", yaxis_title="")
        st.plotly_chart(polish(fig), use_container_width=True)
    with c2:
        mix = filtered["Outcome"].value_counts().reindex(["Stayed", "Left"], fill_value=0).reset_index()
        mix.columns = ["Outcome", "Employees"]
        fig = px.pie(mix, names="Outcome", values="Employees", hole=.64, color="Outcome",
                     color_discrete_map=COLORS, title="Workforce outcome composition")
        fig.update_traces(textinfo="percent+label", textposition="outside", pull=[0, .04])
        fig.update_layout(annotations=[dict(text=f"{n:,}<br>people", x=.5, y=.5, showarrow=False, font_size=18)])
        st.plotly_chart(polish(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        overtime_counts = filtered.groupby(["OverTime", "Outcome"]).size().reset_index(name="Employees")
        fig = px.bar(overtime_counts, x="OverTime", y="Employees", color="Outcome", barmode="group",
                     color_discrete_map=COLORS, text="Employees", title="Overtime and workforce outcomes")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="Overtime status", yaxis_title="Employees")
        st.plotly_chart(polish(fig), use_container_width=True)
    with c2:
        roles = rate_by(filtered, "JobRole").query("Employees >= 5").sort_values("AttritionRate", ascending=False)
        fig = px.bar(roles, x="JobRole", y="AttritionRate", color="AttritionRate", color_continuous_scale="Sunset",
                     hover_data=["Employees", "Departures"], title="Job roles ranked by attrition rate")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Attrition rate (%)")
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(polish(fig, 460, True), use_container_width=True)

with drivers:
    st.subheader("Explore patterns behind employee departures")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.violin(filtered, x="Outcome", y="MonthlyIncome", color="Outcome", box=True, points=False,
                        color_discrete_map=COLORS, title="Monthly income distribution by outcome")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monthly income (dataset units)")
        st.plotly_chart(polish(fig), use_container_width=True)
    with c2:
        sat = rate_by(filtered, "JobSatisfaction").sort_values("JobSatisfaction")
        fig = px.line(sat, x="JobSatisfaction", y="AttritionRate", markers=True,
                      title="Attrition rate across job satisfaction levels", custom_data=["Employees"])
        fig.update_traces(line=dict(color="#8B5CF6", width=4), marker=dict(size=12, color="#F59E0B"),
                          hovertemplate="Satisfaction: %{x}<br>Attrition: %{y:.1f}%<br>Employees: %{customdata[0]}<extra></extra>")
        fig.update_layout(xaxis_title="Job satisfaction rating", yaxis_title="Attrition rate (%)")
        st.plotly_chart(polish(fig, percent=True), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        tenure = filtered.copy()
        tenure["TenureBand"] = pd.cut(tenure["YearsAtCompany"], bins=[-1, 1, 3, 5, 10, 20, 100],
                                      labels=["0–1", "2–3", "4–5", "6–10", "11–20", "21+"])
        bands = rate_by(tenure, "TenureBand")
        fig = px.area(bands, x="TenureBand", y="AttritionRate", markers=True,
                      title="Attrition across company tenure bands")
        fig.update_traces(line=dict(color="#06B6D4", width=3), fillcolor="rgba(6,182,212,.20)")
        fig.update_layout(xaxis_title="Years at company", yaxis_title="Attrition rate (%)")
        st.plotly_chart(polish(fig, percent=True), use_container_width=True)
    with c2:
        heat = filtered.pivot_table(index="JobSatisfaction", columns="OverTime", values="AttritionFlag", aggfunc="mean") * 100
        fig = px.imshow(heat, text_auto=".1f", color_continuous_scale="RdPu", aspect="auto",
                        labels=dict(x="Overtime", y="Job satisfaction", color="Attrition %"),
                        title="Attrition heatmap: satisfaction × overtime")
        fig.update_layout(coloraxis_colorbar_ticksuffix="%")
        st.plotly_chart(polish(fig), use_container_width=True)

    feature = st.selectbox("Compare another employee characteristic", [c for c in ["BusinessTravel", "WorkLifeBalance", "EnvironmentSatisfaction", "JobLevel", "MaritalStatus", "JobRole"] if c in filtered.columns])
    breakdown = rate_by(filtered, feature).sort_values("AttritionRate", ascending=False)
    fig = px.scatter(breakdown, x=feature, y="AttritionRate", size="Employees", color="AttritionRate",
                     color_continuous_scale="Turbo", size_max=48, hover_data=["Employees", "Departures"],
                     title=f"Attrition rate and group size by {feature}")
    fig.update_layout(xaxis_title="", yaxis_title="Attrition rate (%)")
    st.plotly_chart(polish(fig, percent=True), use_container_width=True)
    st.dataframe(breakdown.style.format({"AttritionRate": "{:.1f}%"}), use_container_width=True, hide_index=True)
    st.caption("Smaller groups produce less stable rates. Monthly income is shown in source units because currency is not documented.")

with model_tab:
    st.subheader("Holdout-tested logistic regression")
    st.caption("Trained on the complete uploaded dataset, independently of dashboard filters. Exploratory research only; not for employee-level employment decisions.")
    if st.button("Train and evaluate model", type="primary"):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                                         precision_recall_curve, confusion_matrix, classification_report)
            cols = [c for c in ["Age", "BusinessTravel", "Department", "DistanceFromHome", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "OverTime", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"] if c in cleaned.columns]
            X, y = cleaned[cols], cleaned["AttritionFlag"]
            numeric = X.select_dtypes(include="number").columns.tolist()
            categorical = [c for c in cols if c not in numeric]
            prep = ColumnTransformer([("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                                      ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
            pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
            pipe.fit(Xtrain, ytrain)
            scores = pipe.predict_proba(Xtest)[:, 1]
            predictions = pipe.predict(Xtest)
            auc, ap = roc_auc_score(ytest, scores), average_precision_score(ytest, scores)
            m1, m2, m3 = st.columns(3)
            m1.metric("ROC-AUC", f"{auc:.3f}")
            m2.metric("Average precision", f"{ap:.3f}")
            m3.metric("Holdout employees", f"{len(ytest):,}")
            c1, c2 = st.columns(2)
            with c1:
                fpr, tpr, _ = roc_curve(ytest, scores)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#6366F1", width=4), name="Model"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94A3B8", dash="dash"), name="Random baseline"))
                fig.update_layout(title=f"ROC curve (AUC {auc:.3f})", xaxis_title="False positive rate", yaxis_title="True positive rate")
                st.plotly_chart(polish(fig), use_container_width=True)
            with c2:
                precision, recall, _ = precision_recall_curve(ytest, scores)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color="#EC4899", width=4), name="Model"))
                fig.add_hline(y=float(ytest.mean()), line_dash="dash", line_color="#94A3B8", annotation_text="Class prevalence")
                fig.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(polish(fig), use_container_width=True)
            cm = confusion_matrix(ytest, predictions)
            fig = px.imshow(cm, text_auto=True, x=["Predicted stayed", "Predicted left"],
                            y=["Actual stayed", "Actual left"], color_continuous_scale="Blues", title="Confusion matrix")
            st.plotly_chart(polish(fig, 340), use_container_width=True)
            st.text(classification_report(ytest, predictions, zero_division=0))
            st.caption("25% stratified test split; synthetic cross-sectional data. No hyperparameter tuning or real-world validation. Class-weighted model uses a 0.5 classification threshold.")
        except Exception as exc:
            st.error(f"Model evaluation failed: {type(exc).__name__}: {exc}")

with briefing:
    dept_rates = filtered.groupby("Department")["AttritionFlag"].agg(["mean", "size"]).round(4).to_dict("index")
    overtime_rates = filtered.groupby("OverTime")["AttritionFlag"].agg(["mean", "size"]).round(4).to_dict("index")
    facts = {"dataset": "IBM HR Analytics fictional employee dataset", "employee_records": n,
             "attrition_cases": left, "attrition_rate_pct": round(rate, 2),
             "department_rates": dept_rates, "overtime_rates": overtime_rates,
             "limitations": ["fictional data", "cross-sectional observations", "association does not imply causation", "income currency undocumented", "dashboard filters affect these figures"]}
    st.subheader("Evidence-based executive briefing")
    st.write(f"Among {n:,} fictional employee records matching the selected filters, {left:,} are labeled as departures ({rate:.1f}%).")
    st.dataframe(pd.DataFrame([{"Department": k, "Employees": v["size"], "Attrition rate (%)": round(v["mean"] * 100, 1)} for k, v in dept_rates.items()]), hide_index=True, use_container_width=True)
    if use_ai:
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Set the OPENAI_API_KEY environment variable to enable the AI briefing.")
        elif st.button("Generate AI executive briefing"):
            try:
                from openai import OpenAI
                response = OpenAI().responses.create(model=model, input="You are an HR analytics consultant. Give four findings and three suggested next analyses based STRICTLY on this JSON. Explicitly mention synthetic data, subgroup-size uncertainty, and noncausal associations. Do not make employee-level recommendations or invent data.\n" + json.dumps(facts))
                st.write(response.output_text)
            except Exception as exc:
                st.error(f"AI briefing unavailable: {type(exc).__name__}")
    st.download_button("Download aggregated findings (JSON)", json.dumps(facts, indent=2), "retention_findings.json", "application/json")
    st.download_button("Download cleaned data (CSV)", cleaned.to_csv(index=False), "hr_attrition_cleaned.csv", "text/csv")

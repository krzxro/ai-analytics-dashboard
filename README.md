# InsightPilot — AI-Powered Employee Retention Analytics

An interactive portfolio project using the **IBM HR Analytics Employee Attrition & Performance** dataset published on Kaggle. It includes exploratory business intelligence, automated cleaning, a holdout-evaluated logistic regression model, and optional OpenAI-generated executive summaries.

## Dataset (required; not fabricated)

Download the dataset from https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset and extract `WA_Fn-UseC_-HR-Employee-Attrition.csv`. Either upload it in the sidebar or place it in `data/` next to `app.py`.

The data is **fictional**, not actual IBM workforce data.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: set `OPENAI_API_KEY` in your environment to enable LLM-generated summaries. This sends only aggregate data to the model, not individual employee rows, and API use may cost money.

## Features

- Attrition rates by department, job role, overtime, satisfaction, and more
- Interactive charts and breakdowns with group sizes
- Data cleaning audit
- Logistic regression with a stratified 75/25 train/test split, ROC-AUC and average precision
- Grounded generative AI executive briefing using aggregate JSON
- CSV and JSON exports

## Limitations

- Observational and fictional data: do not infer causal effects or actual employee risk.
- Dataset contains no dates: do not invent time-series trends.
- No currency is documented for income columns.
- The classifier is a demonstration only; not for actual hiring or HR decisions.
- A single holdout split is not production-level model validation.


## Attribution

Dataset: IBM HR Analytics Employee Attrition & Performance, Kaggle (pavansubhasht), fictional dataset created by IBM data scientists. Consult the Kaggle dataset license and terms before redistributing raw data.

## Updated dashboard visualizations
The dashboard uses a clean interface with no decorative icons or emojis. The executive overview includes a color-coded horizontal department ranking, donut composition, grouped overtime bars, and a job-role comparison. Retention drivers include income violin plots, satisfaction trend, tenure area chart, satisfaction-by-overtime heatmap, and a bubble chart. Predictive modeling adds ROC and precision–recall curves plus a confusion matrix. Charts respond to sidebar filters (except the model, which always trains on the full dataset).

To update an existing installation, replace its `app.py` with the new file, or extract this entire ZIP to a new folder and run `python -m streamlit run app.py` from the folder containing `requirements.txt`.

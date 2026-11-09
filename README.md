# AI-Powered Employee Retention Analytics Dashboard

An interactive business intelligence and machine learning dashboard built with Python and Streamlit to explore employee attrition, identify workforce patterns, and demonstrate predictive analytics.

The project combines exploratory data analysis, automated preprocessing, logistic regression, and optional generative AI to translate workforce data into actionable business insights.

## Project Overview

Using the IBM HR Analytics Employee Attrition & Performance dataset, this project analyzes 1,470 fictional employee records to explore relationships between attrition and factors such as job satisfaction, overtime, compensation, department, and tenure.

The dashboard is designed to make analytical findings accessible to both technical and non-technical audiences.

## Key Features

**Exploratory Analytics**
- Interactive visualizations of employee attrition across departments, job roles, overtime status, and satisfaction levels
- Workforce comparisons with group sizes and attrition rates
- Compensation and tenure distributions
- Dynamic sidebar filtering

**Predictive Modeling**
- Logistic regression classifier for demonstrating employee attrition prediction
- Stratified 75/25 training and testing split
- Model evaluation using ROC-AUC, average precision, and confusion matrix
- ROC and precision–recall visualizations

**Generative AI Integration**
- Optional OpenAI API integration for generating executive summaries
- Business-oriented interpretations based on aggregate workforce statistics
- No individual employee records transmitted to the API

**Data Processing and Reporting**
- Automated data cleaning and validation
- Data quality audit
- Downloadable CSV and JSON exports

## Dashboard Visualizations

The dashboard includes three primary analytical areas:

**Executive Overview:** Department attrition rankings, workforce composition, overtime comparisons, and job-role analysis.

**Retention Drivers:** Income distributions, job satisfaction patterns, tenure comparisons, satisfaction-by-overtime heatmaps, and multivariable bubble charts.

**Predictive Modeling:** ROC curve, precision–recall curve, and confusion matrix.

Dashboard visualizations respond to sidebar filters, while predictive modeling uses a consistent train/test split of the complete dataset.

## Dataset

This project uses the [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), available on Kaggle.

The dataset is fictional and does not represent actual IBM employees.

To use the application, download `WA_Fn-UseC_-HR-Employee-Attrition.csv` and either upload it through the Streamlit sidebar or place it in the `data/` directory.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

Optionally, configure an `OPENAI_API_KEY` environment variable to enable AI-generated executive summaries. API usage may incur costs.

## Limitations

- The dataset is fictional and observational; findings should not be interpreted as causal relationships.
- The dataset does not contain dates, so calendar-based trend analysis is not supported.
- Income variables have no documented currency.
- The predictive model is intended for educational demonstration and is not suitable for real-world employment decisions.
- A single holdout evaluation does not establish production-level model reliability.

## Data Attribution

IBM HR Analytics Employee Attrition & Performance dataset, published on Kaggle by pavansubhasht. Refer to the dataset's license and usage terms before redistributing the original data.

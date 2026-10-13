"""Deterministic, testable analytics functions; no API required."""
import re
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame):
    df = df.copy()
    original_rows = len(df)
    df.columns = [str(c).strip() for c in df.columns]
    # Make repeated/blank headers unique without discarding columns.
    names, used = [], set()
    for i, name in enumerate(df.columns):
        base = name or f'column_{i+1}'
        candidate, n = base, 2
        while candidate in used:
            candidate = f'{base}_{n}'
            n += 1
        names.append(candidate)
        used.add(candidate)
    df.columns = names
    conversions = []
    for col in df.select_dtypes(include='object').columns:
        values = df[col].astype('string').str.strip()
        parsed = pd.to_numeric(values.str.replace(',', '', regex=False).str.replace('$', '', regex=False), errors='coerce')
        nonnull = values.notna().sum()
        if nonnull and parsed.notna().sum() / nonnull >= 0.95 and parsed.notna().sum() > 0:
            df[col] = parsed
            conversions.append(col)
        else:
            df[col] = values
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {'original_rows': original_rows, 'cleaned_rows': len(df),
                'duplicates_removed': original_rows - len(df), 'converted_to_numeric': conversions,
                'missing_cells': int(df.isna().sum().sum())}


def analyze(df: pd.DataFrame):
    nums = df.select_dtypes(include='number')
    numeric = {}
    for col in nums.columns:
        s = nums[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
            numeric[col] = {'count': int(s.count()), 'mean': round(float(s.mean()), 3),
                            'median': round(float(s.median()), 3), 'min': round(float(s.min()), 3),
                            'max': round(float(s.max()), 3), 'total': round(float(s.sum()), 3),
                            'iqr_outliers': outliers}
    corr_pairs = []
    if len(nums.columns) >= 2:
        corr = nums.corr()
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i+1:]:
                v = corr.loc[c1, c2]
                if pd.notna(v):
                    corr_pairs.append({'columns': [c1, c2], 'correlation': round(float(v), 3)})
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return {'rows': len(df), 'columns': len(df.columns), 'missing_by_column': {str(k): int(v) for k, v in df.isna().sum().items()},
            'numeric': numeric, 'top_correlations': corr_pairs[:10]}


def build_summary(report):
    lines = [f"Analyzed {report['rows']:,} rows across {report['columns']} columns."]
    missing = [(k, v) for k, v in report['missing_by_column'].items() if v]
    if missing:
        name, count = max(missing, key=lambda x: x[1])
        lines.append(f'Missing data detected in {len(missing)} column(s); {name} has the most ({count} cells).')
    else:
        lines.append('No missing values detected.')
    for name, stats in list(report['numeric'].items())[:3]:
        lines.append(f'{name}: mean {stats["mean"]:,.2f}, median {stats["median"]:,.2f}, IQR outliers {stats["iqr_outliers"]}.')
    if report['top_correlations']:
        top = report['top_correlations'][0]
        lines.append(f'Strongest observed linear correlation: {top["columns"][0]} and {top["columns"][1]} (r={top["correlation"]:.2f}); correlation does not imply causation.')
    return lines


def answer_question(question: str, df: pd.DataFrame):
    q = question.lower().strip()
    if any(x in q for x in ['how many rows', 'number of rows', 'record count', 'how many records']):
        return f'There are {len(df):,} rows.'
    if 'missing' in q or 'null' in q:
        counts = df.isna().sum()
        return 'Missing values: ' + ', '.join(f'{k}: {v}' for k, v in counts.items() if v > 0) if counts.sum() else 'No missing values detected.'
    numeric = df.select_dtypes(include='number').columns.tolist()
    mentions = [col for col in numeric if re.search(r'(?<!\w)' + re.escape(col.lower()) + r'(?!\w)', q)]
    if 'correlation' in q or 'correlat' in q:
        if len(mentions) >= 2:
            pair = df[mentions[:2]].corr().iloc[0, 1]
            return f'Pearson correlation between {mentions[0]} and {mentions[1]}: {pair:.3f}.' if pd.notna(pair) else 'Correlation is undefined for these columns.'
        return 'Please mention two numeric column names to calculate their correlation.'
    if mentions:
        col = mentions[0]
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return f'{col} has no valid numeric values.'
        if any(x in q for x in ['average', 'mean']):
            return f'Average {col}: {s.mean():,.2f}.'
        if any(x in q for x in ['total', 'sum']):
            return f'Total {col}: {s.sum():,.2f}.'
        if any(x in q for x in ['minimum', 'lowest', 'min']):
            return f'Minimum {col}: {s.min():,.2f}.'
        if any(x in q for x in ['maximum', 'highest', 'max']):
            return f'Maximum {col}: {s.max():,.2f}.'
    return 'Try asking about row count, missing values, average/total/min/max of a numeric column, or correlation between two numeric columns.'

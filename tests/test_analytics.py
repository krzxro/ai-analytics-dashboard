import pandas as pd
from analytics import clean_data, analyze, answer_question, build_summary


def test_clean_and_analyze():
    raw = pd.DataFrame({' revenue ': ['$10', '$20', '$20'], 'cost': [5, 10, 10]})
    clean, audit = clean_data(raw)
    assert len(clean) == 2
    assert audit['duplicates_removed'] == 1
    assert clean['revenue'].sum() == 30
    report = analyze(clean)
    assert report['numeric']['revenue']['mean'] == 15
    assert len(build_summary(report)) >= 2
    assert '30.00' in answer_question('total revenue', clean)
    assert '2 rows' in answer_question('how many rows', clean)


def test_missing():
    df = pd.DataFrame({'a': [1, None], 'b': [3, 4]})
    assert 'a: 1' in answer_question('missing values', df)

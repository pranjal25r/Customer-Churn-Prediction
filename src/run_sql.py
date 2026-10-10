"""
Run the SQL churn analysis and save the results as Markdown.

Loads the CSV into an in-memory SQLite table called "customers", runs every
query in sql/churn_analysis.sql and writes the result tables to
reports/churn_rates.md. Also lists the chosen model's top features as a
cross-check against the churn-rate tables.

Run with: python -m src.run_sql
"""

import sqlite3
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.data_processing import load_data
from src.predict import load_model, top_features


ROOT_DIR = Path(__file__).parent.parent
SQL_PATH = ROOT_DIR / 'sql' / 'churn_analysis.sql'
REPORT_PATH = ROOT_DIR / 'reports' / 'churn_rates.md'

TITLE_PREFIX = '-- title:'


def read_queries(filepath: Path = SQL_PATH) -> List[Tuple[str, str]]:
    """
    Split the SQL file into (title, query) pairs.

    Args:
        filepath: Path to the SQL file.

    Returns:
        List of (title, query) tuples in file order.
    """
    queries = []
    for block in filepath.read_text().split(TITLE_PREFIX)[1:]:
        title, _, query = block.partition('\n')
        queries.append((title.strip(), query.strip()))
    return queries


def to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a Markdown table."""
    lines = [
        '| ' + ' | '.join(df.columns) + ' |',
        '|' + '|'.join('---' for _ in df.columns) + '|'
    ]
    for row in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def run_sql_analysis() -> str:
    """
    Run all queries and write the report.

    Returns:
        The report text.
    """
    conn = sqlite3.connect(':memory:')
    load_data().to_sql('customers', conn, index=False)

    sections = [
        '# Churn rates',
        'Produced by `python -m src.run_sql` from `sql/churn_analysis.sql`, '
        'using all 7,043 customers in the CSV.'
    ]
    for title, query in read_queries():
        sections.append(f'## {title}')
        sections.append(to_markdown(pd.read_sql_query(query, conn)))
    conn.close()

    try:
        pipeline = load_model()
    except FileNotFoundError:
        print("Model not found, skipping top features. Run train_model.py first.")
    else:
        features = top_features(pipeline, top_n=5)
        features['weight'] = features['weight'].round(3)
        model_name = type(pipeline.named_steps['model']).__name__
        sections.append("## Cross-check: the model's top 5 features")
        sections.append(
            f'The 5 inputs the chosen model ({model_name}) relies on most. '
            'Names are the columns after one-hot encoding, e.g. '
            '`Contract_Month-to-month` means "Contract is Month-to-month".'
        )
        sections.append(to_markdown(features))

    report = '\n\n'.join(sections) + '\n'
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    return report


if __name__ == '__main__':
    print(run_sql_analysis())
    print(f"Saved to: {REPORT_PATH}")

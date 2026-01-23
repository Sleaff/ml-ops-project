"""Data drift detection using Evidently AI."""

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def load_reference_data(path: str = "data/processed/news.csv") -> pd.DataFrame:
    """Load reference (training) data."""
    df = pd.read_csv(path)
    return df


def create_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
) -> Report:
    """Generate drift report comparing reference and current data."""
    column_mapping = ColumnMapping(
        target="label",
    )

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

    return report


if __name__ == "__main__":
    print("Loading reference data...")
    full_data = load_reference_data()

    # Use smaller samples for report (full dataset is too large for HTML)
    print("Sampling data for report...")
    reference = full_data.sample(n=500, random_state=42)
    current = full_data.sample(n=500, random_state=123)

    print("Generating drift report...")
    create_drift_report(reference, current)
    print("Done! Open reports/drift_report.html in your browser.")

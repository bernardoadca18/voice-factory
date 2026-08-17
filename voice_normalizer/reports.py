"""Report generation for batch processing."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import BatchReport, FileResult


def save_json_report(report: BatchReport, output_path: Path) -> Path:
    """
    Save a batch report as JSON.

    Args:
        report: Batch report to save.
        output_path: Path for the output file.

    Returns:
        Path to the saved report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp to report
    report_data = report.to_dict()
    report_data["timestamp"] = datetime.now().isoformat()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return output_path


def save_csv_report(report: BatchReport, output_path: Path) -> Path:
    """
    Save a batch report as CSV.

    Args:
        report: Batch report to save.
        output_path: Path for the output file.

    Returns:
        Path to the saved report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    fieldnames = [
        "file",
        "status",
        "input_lufs",
        "output_lufs",
        "input_true_peak",
        "output_true_peak",
        "applied_gain_db",
        "clipping_before",
        "clipping_after",
        "warnings",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for file_result in report.files:
            row = {
                "file": file_result.file,
                "status": file_result.status.value,
                "input_lufs": (
                    round(file_result.before.lufs, 2) if file_result.before else ""
                ),
                "output_lufs": (
                    round(file_result.after.lufs, 2) if file_result.after else ""
                ),
                "input_true_peak": (
                    round(file_result.before.true_peak_db, 2)
                    if file_result.before
                    else ""
                ),
                "output_true_peak": (
                    round(file_result.after.true_peak_db, 2) if file_result.after else ""
                ),
                "applied_gain_db": (
                    round(file_result.processing.applied_gain_db, 2)
                    if file_result.processing
                    else ""
                ),
                "clipping_before": file_result.before.clipping if file_result.before else "",
                "clipping_after": file_result.after.clipping if file_result.after else "",
                "warnings": "; ".join(file_result.warnings) if file_result.warnings else "",
                "error": file_result.error or "",
            }
            writer.writerow(row)

    return output_path


def generate_reports(report: BatchReport, output_dir: Path) -> tuple[Path, Optional[Path]]:
    """
    Generate JSON and CSV reports for a batch.

    Args:
        report: Batch report.
        output_dir: Directory for report files.

    Returns:
        Tuple of (json_path, csv_path).
    """
    json_path = save_json_report(report, output_dir / "report.json")
    csv_path = save_csv_report(report, output_dir / "report.csv")

    return json_path, csv_path

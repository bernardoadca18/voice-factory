"""Data models for the voice normalizer."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class FileStatus(str, Enum):
    """Status of a processed file."""
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class AudioAnalysis:
    """Audio analysis results for a WAV file."""
    filename: str
    file_path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    lufs: float
    sample_peak_db: float
    true_peak_db: float
    clipping: bool
    rms_db: Optional[float] = None
    dc_offset: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "file_path": str(self.file_path),
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "lufs": round(self.lufs, 2),
            "sample_peak_db": round(self.sample_peak_db, 2),
            "true_peak_db": round(self.true_peak_db, 2),
            "clipping": self.clipping,
            "rms_db": round(self.rms_db, 2) if self.rms_db is not None else None,
            "dc_offset": round(self.dc_offset, 6) if self.dc_offset is not None else None,
        }


@dataclass
class GainCalculation:
    """Results of gain calculation."""
    gain_for_target_db: float
    gain_allowed_by_peak_db: float
    applied_gain_db: float
    target_reached: bool
    peak_limited: bool
    gain_capped: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gain_for_target_db": round(self.gain_for_target_db, 2),
            "gain_allowed_by_peak_db": round(self.gain_allowed_by_peak_db, 2),
            "applied_gain_db": round(self.applied_gain_db, 2),
            "target_reached": self.target_reached,
            "peak_limited": self.peak_limited,
            "gain_capped": self.gain_capped,
        }


@dataclass
class FileResult:
    """Complete processing result for a single file."""
    file: str
    input_path: str
    output_path: Optional[str]
    before: Optional[AudioAnalysis]
    processing: Optional[GainCalculation]
    after: Optional[AudioAnalysis]
    status: FileStatus
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "file": self.file,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "before": self.before.to_dict() if self.before else None,
            "processing": self.processing.to_dict() if self.processing else None,
            "after": self.after.to_dict() if self.after else None,
            "status": self.status.value,
            "warnings": self.warnings,
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class BatchSummary:
    """Summary of a batch processing run."""
    total_files: int
    processed: int
    passed: int
    warnings: int
    errors: int
    skipped: int
    average_lufs_before: Optional[float]
    average_lufs_after: Optional[float]
    maximum_true_peak_after: Optional[float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "processed": self.processed,
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "skipped": self.skipped,
            "average_lufs_before": round(self.average_lufs_before, 2) if self.average_lufs_before is not None else None,
            "average_lufs_after": round(self.average_lufs_after, 2) if self.average_lufs_after is not None else None,
            "maximum_true_peak_after": round(self.maximum_true_peak_after, 2) if self.maximum_true_peak_after is not None else None,
        }


@dataclass
class BatchReport:
    """Complete batch processing report."""
    input_dir: str
    output_dir: str
    config: dict
    files: list[FileResult]
    summary: BatchSummary

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "config": self.config,
            "files": [f.to_dict() for f in self.files],
            "summary": self.summary.to_dict(),
        }

"""Configuration management for the voice normalizer."""

from pathlib import Path
from typing import Optional

import yaml


class Config:
    """Configuration container for normalization settings."""

    def __init__(
        self,
        target_lufs: float = -20.0,
        max_true_peak_db: float = -3.0,
        max_positive_gain_db: float = 6.0,
        acceptable_lufs_min: float = -22.0,
        acceptable_lufs_max: float = -18.0,
        reject_source_clipping: bool = True,
        overwrite_output: bool = False,
    ):
        self.target_lufs = target_lufs
        self.max_true_peak_db = max_true_peak_db
        self.max_positive_gain_db = max_positive_gain_db
        self.acceptable_lufs_min = acceptable_lufs_min
        self.acceptable_lufs_max = acceptable_lufs_max
        self.reject_source_clipping = reject_source_clipping
        self.overwrite_output = overwrite_output

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            return cls()

        norm = data.get("normalization", {})
        val = data.get("validation", {})
        proc = data.get("processing", {})

        return cls(
            target_lufs=norm.get("target_lufs", -20.0),
            max_true_peak_db=norm.get("max_true_peak_db", -3.0),
            max_positive_gain_db=norm.get("max_positive_gain_db", 6.0),
            acceptable_lufs_min=val.get("acceptable_lufs_min", -22.0),
            acceptable_lufs_max=val.get("acceptable_lufs_max", -18.0),
            reject_source_clipping=val.get("reject_source_clipping", True),
            overwrite_output=proc.get("overwrite_output", False),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "normalization": {
                "target_lufs": self.target_lufs,
                "max_true_peak_db": self.max_true_peak_db,
                "max_positive_gain_db": self.max_positive_gain_db,
            },
            "validation": {
                "acceptable_lufs_min": self.acceptable_lufs_min,
                "acceptable_lufs_max": self.acceptable_lufs_max,
                "reject_source_clipping": self.reject_source_clipping,
            },
            "processing": {
                "overwrite_output": self.overwrite_output,
            },
        }

    def __repr__(self) -> str:
        return (
            f"Config(target_lufs={self.target_lufs}, "
            f"max_true_peak_db={self.max_true_peak_db}, "
            f"max_positive_gain_db={self.max_positive_gain_db})"
        )

"""Gain calculation and normalization logic."""

from pathlib import Path
from typing import Optional

from .analyzer import Analyzer
from .config import Config
from .models import AudioAnalysis, GainCalculation


class Normalizer:
    """Calculates and applies linear gain for normalization."""

    def __init__(self, config: Config):
        """
        Initialize the normalizer.

        Args:
            config: Configuration for normalization parameters.
        """
        self.config = config
        self.analyzer = Analyzer()

    def calculate_gain(self, analysis: AudioAnalysis) -> GainCalculation:
        """
        Calculate the appropriate gain for normalization.

        The gain is calculated to:
        1. Move toward target LUFS
        2. Respect maximum true peak (with safety margin)
        3. Respect maximum positive gain

        Args:
            analysis: Audio analysis results.

        Returns:
            GainCalculation with gain values and constraints.
        """
        # Calculate gain needed to reach target LUFS
        gain_for_target_db = self.config.target_lufs - analysis.lufs

        # Calculate maximum gain allowed by true peak constraint
        # Use the effective max (with safety margin) for processing
        # We want: final_true_peak <= effective_max_true_peak_db
        # So: analysis.true_peak_db + gain <= effective_max_true_peak_db
        # Therefore: gain <= effective_max_true_peak_db - analysis.true_peak_db
        effective_peak_ceiling = self.config.effective_max_true_peak_db
        gain_allowed_by_peak_db = effective_peak_ceiling - analysis.true_peak_db

        # Start with the gain needed for target
        applied_gain_db = gain_for_target_db

        # Track constraints
        peak_limited = False
        gain_capped = False
        target_reached = True

        # Check if positive gain is limited by peak constraint
        if applied_gain_db > gain_allowed_by_peak_db:
            applied_gain_db = gain_allowed_by_peak_db
            peak_limited = True
            target_reached = False

        # Check if gain exceeds maximum positive gain
        if applied_gain_db > self.config.max_positive_gain_db:
            applied_gain_db = self.config.max_positive_gain_db
            gain_capped = True
            target_reached = False

        # Negative gain (reduction) is always allowed
        # No need to cap negative gain

        return GainCalculation(
            gain_for_target_db=gain_for_target_db,
            gain_allowed_by_peak_db=gain_allowed_by_peak_db,
            applied_gain_db=applied_gain_db,
            target_reached=target_reached,
            peak_limited=peak_limited,
            gain_capped=gain_capped,
        )

    def check_gain_excessive(self, calculation: GainCalculation) -> bool:
        """
        Check if the required gain is excessive.

        This happens when gain_for_target_db exceeds max_positive_gain_db
        significantly, indicating a very quiet file.

        Args:
            calculation: Gain calculation results.

        Returns:
            True if gain is excessive and requires manual review.
        """
        return calculation.gain_for_target_db > self.config.max_positive_gain_db

    def calculate_linear_gain(self, gain_db: float) -> float:
        """
        Convert dB gain to linear gain multiplier.

        Args:
            gain_db: Gain in decibels.

        Returns:
            Linear gain multiplier.
        """
        return 10 ** (gain_db / 20.0)


def calculate_normalization_gain(
    analysis: AudioAnalysis,
    target_lufs: float,
    max_true_peak_db: float,
    max_positive_gain_db: float,
) -> GainCalculation:
    """
    Convenience function to calculate normalization gain.

    Args:
        analysis: Audio analysis results.
        target_lufs: Target loudness in LUFS.
        max_true_peak_db: Maximum true peak in dBTP.
        max_positive_gain_db: Maximum positive gain in dB.

    Returns:
        GainCalculation with gain values.
    """
    config = Config(
        target_lufs=target_lufs,
        max_true_peak_db=max_true_peak_db,
        max_positive_gain_db=max_positive_gain_db,
    )
    normalizer = Normalizer(config)
    return normalizer.calculate_gain(analysis)

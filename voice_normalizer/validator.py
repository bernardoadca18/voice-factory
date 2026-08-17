"""Validation logic for normalized audio files."""

from pathlib import Path
from typing import Optional

from .config import Config
from .models import AudioAnalysis, FileStatus


class Validator:
    """Validates audio files against quality criteria."""

    def __init__(self, config: Config):
        """
        Initialize the validator.

        Args:
            config: Configuration for validation parameters.
        """
        self.config = config

    def validate_source(self, analysis: AudioAnalysis) -> tuple[bool, list[str]]:
        """
        Validate source audio before processing.

        Args:
            analysis: Audio analysis of the source file.

        Returns:
            Tuple of (is_valid, warnings) - is_valid=False means ERROR status.
        """
        warnings = []

        # Check for clipping
        if analysis.clipping:
            if self.config.reject_source_clipping:
                return False, ["Source audio has clipping - cannot process safely"]
            else:
                warnings.append("Source audio has clipping - quality may be degraded")

        # Check for unusual format
        if analysis.channels > 2:
            warnings.append(f"Unusual channel count: {analysis.channels}")

        if analysis.sample_rate < 16000:
            warnings.append(f"Low sample rate: {analysis.sample_rate} Hz")

        if analysis.sample_rate > 96000:
            warnings.append(f"High sample rate: {analysis.sample_rate} Hz")

        # Check for very short files
        if analysis.duration_seconds < 0.5:
            warnings.append(f"Very short file: {analysis.duration_seconds:.2f}s")

        return True, warnings

    def validate_normalized_file(
        self,
        analysis: AudioAnalysis,
    ) -> tuple[FileStatus, list[str]]:
        """
        Validate a single normalized file without before/after comparison.

        Used by the 'validate' CLI command to check already-processed files.

        Args:
            analysis: Audio analysis of the file.

        Returns:
            Tuple of (status, warnings).
        """
        warnings = []

        # Check LUFS is in acceptable range
        lufs_ok = self._check_lufs(analysis, warnings)

        # Check true peak constraint with tolerance
        peak_ok = self._check_true_peak(analysis, warnings)

        # Check for clipping
        clipping_ok = self._check_clipping(analysis, warnings)

        # Determine final status using same logic as validate_result
        # ERROR: Real problems that make the file unusable
        # WARNING: File is usable but has notable conditions
        # PASS: Everything is good
        if not lufs_ok or not peak_ok or not clipping_ok:
            status = FileStatus.ERROR
        elif warnings:
            status = FileStatus.WARNING
        else:
            status = FileStatus.PASS

        return status, warnings

    def validate_result(
        self,
        before: AudioAnalysis,
        after: AudioAnalysis,
        gain_calculation: Optional,
    ) -> tuple[FileStatus, list[str]]:
        """
        Validate the normalization result.

        Args:
            before: Analysis before processing.
            after: Analysis after processing.
            gain_calculation: The gain calculation that was applied.

        Returns:
            Tuple of (status, warnings).
        """
        warnings = []

        # Check LUFS is in acceptable range
        lufs_ok = self._check_lufs(after, warnings)

        # Check true peak constraint with tolerance for floating point comparison
        peak_ok = self._check_true_peak(after, warnings)

        # Check for clipping introduced by processing
        clipping_ok = self._check_clipping_introduced(before, after, warnings)

        # Check if target was not reached due to constraints
        # This is an expected condition, not an error
        if gain_calculation and gain_calculation.peak_limited:
            warnings.append(
                "Target LUFS not reached because True Peak constraint had priority. "
                "Output remains within acceptable loudness range."
            )

        if gain_calculation and gain_calculation.gain_capped:
            warnings.append(
                f"Gain limited to {self.config.max_positive_gain_db} dB "
                f"(required: {gain_calculation.gain_for_target_db:.2f} dB) - "
                "manual review recommended"
            )

        # Determine final status
        # ERROR: Real problems that make the file unusable
        # WARNING: File is usable but has notable conditions
        # PASS: Everything is good
        if not lufs_ok or not peak_ok or not clipping_ok:
            status = FileStatus.ERROR
        elif warnings:
            status = FileStatus.WARNING
        else:
            status = FileStatus.PASS

        return status, warnings

    def _check_lufs(self, analysis: AudioAnalysis, warnings: list[str]) -> bool:
        """
        Check if LUFS is in acceptable range.

        Args:
            analysis: Audio analysis.
            warnings: List to append warnings to.

        Returns:
            True if LUFS is acceptable.
        """
        lufs_ok = (
            self.config.acceptable_lufs_min <= analysis.lufs <= self.config.acceptable_lufs_max
        )

        if not lufs_ok:
            warnings.append(
                f"LUFS {analysis.lufs:.2f} outside acceptable range "
                f"[{self.config.acceptable_lufs_min}, {self.config.acceptable_lufs_max}]"
            )

        return lufs_ok

    def _check_true_peak(self, analysis: AudioAnalysis, warnings: list[str]) -> bool:
        """
        Check if True Peak is within limits with tolerance.

        Args:
            analysis: Audio analysis.
            warnings: List to append warnings to.

        Returns:
            True if True Peak is acceptable.
        """
        # Check true peak constraint with tolerance for floating point comparison
        # Using tolerance prevents false positives from floating point rounding
        peak_exceeded_threshold = (
            analysis.true_peak_db > self.config.max_true_peak_db + self.config.true_peak_tolerance_db
        )

        if peak_exceeded_threshold:
            warnings.append(
                f"True peak {analysis.true_peak_db:.2f} dBTP significantly exceeds maximum "
                f"{self.config.max_true_peak_db} dBTP (tolerance: {self.config.true_peak_tolerance_db} dB)"
            )
            return False

        return True

    def _check_clipping(self, analysis: AudioAnalysis, warnings: list[str]) -> bool:
        """
        Check if audio has clipping.

        Args:
            analysis: Audio analysis.
            warnings: List to append warnings to.

        Returns:
            True if no clipping detected.
        """
        if analysis.clipping:
            warnings.append("Clipping detected in audio")
            return False
        return True

    def _check_clipping_introduced(
        self,
        before: AudioAnalysis,
        after: AudioAnalysis,
        warnings: list[str],
    ) -> bool:
        """
        Check if clipping was introduced by processing.

        Args:
            before: Analysis before processing.
            after: Analysis after processing.
            warnings: List to append warnings to.

        Returns:
            True if no clipping was introduced.
        """
        clipping_introduced = after.clipping and not before.clipping
        if clipping_introduced:
            warnings.append("Clipping introduced by processing - this indicates a bug")
            return False
        return True

    def check_input_output_paths(
        self,
        input_path: Path,
        output_path: Path,
    ) -> tuple[bool, str]:
        """
        Check that input and output paths are safe.

        Args:
            input_path: Path to input file.
            output_path: Path to output file.

        Returns:
            Tuple of (is_safe, error_message).
        """
        # Resolve to absolute paths
        input_resolved = input_path.resolve()
        output_resolved = output_path.resolve()

        # Check if same file
        if input_resolved == output_resolved:
            return False, "Input and output paths resolve to the same file - refusing to overwrite input"

        # Check if output is inside input directory (when processing directories)
        # This is a safety check to prevent accidental modification of source
        try:
            output_resolved.relative_to(input_resolved.parent)
            # Output is in same directory as input - this is OK if filename differs
            if output_resolved.name == input_resolved.name:
                return False, "Output file would overwrite input file"
        except ValueError:
            # Output is in different directory - this is expected
            pass

        return True, ""

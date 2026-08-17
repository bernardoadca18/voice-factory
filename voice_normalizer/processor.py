"""Batch processor for WAV normalization."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf

from .analyzer import Analyzer
from .config import Config
from .models import (
    AudioAnalysis,
    BatchReport,
    BatchSummary,
    FileResult,
    FileStatus,
    GainCalculation,
)
from .normalizer import Normalizer
from .validator import Validator


class Processor:
    """Processes WAV files in batch with linear gain normalization."""

    def __init__(self, config: Config):
        """
        Initialize the processor.

        Args:
            config: Configuration for processing.
        """
        self.config = config
        self.analyzer = Analyzer()
        self.normalizer = Normalizer(config)
        self.validator = Validator(config)

    def scan_wav_files(self, input_dir: Path) -> list[Path]:
        """
        Scan a directory for WAV files.

        Args:
            input_dir: Directory to scan.

        Returns:
            List of WAV file paths, sorted alphabetically.
        """
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        wav_files = sorted(input_dir.glob("*.wav"))
        wav_files_lower = sorted(input_dir.glob("*.WAV"))
        all_files = sorted(set(wav_files + wav_files_lower))

        return all_files

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> FileResult:
        """
        Process a single WAV file.

        Args:
            input_path: Path to input WAV file.
            output_path: Path to output WAV file.

        Returns:
            FileResult with processing status and analysis.
        """
        # Check path safety
        is_safe, error_msg = self.validator.check_input_output_paths(
            input_path, output_path
        )
        if not is_safe:
            return FileResult(
                file=input_path.name,
                input_path=str(input_path),
                output_path=None,
                before=None,
                processing=None,
                after=None,
                status=FileStatus.ERROR,
                error=error_msg,
            )

        # Analyze input
        try:
            before_analysis = self.analyzer.analyze(input_path)
        except Exception as e:
            return FileResult(
                file=input_path.name,
                input_path=str(input_path),
                output_path=None,
                before=None,
                processing=None,
                after=None,
                status=FileStatus.ERROR,
                error=f"Failed to analyze input: {str(e)}",
            )

        # Validate source
        is_valid, warnings = self.validator.validate_source(before_analysis)
        if not is_valid:
            return FileResult(
                file=input_path.name,
                input_path=str(input_path),
                output_path=None,
                before=before_analysis,
                processing=None,
                after=None,
                status=FileStatus.ERROR,
                warnings=warnings,
                error="Source file rejected",
            )

        # Check if output exists
        if output_path.exists() and not self.config.overwrite_output:
            return FileResult(
                file=input_path.name,
                input_path=str(input_path),
                output_path=str(output_path),
                before=before_analysis,
                processing=None,
                after=None,
                status=FileStatus.SKIPPED,
                warnings=["Output file already exists and overwrite is disabled"],
            )

        # Calculate gain
        gain_calculation = self.normalizer.calculate_gain(before_analysis)

        # Check for excessive gain requirement
        if self.normalizer.check_gain_excessive(gain_calculation):
            warnings.append(
                f"Required gain {gain_calculation.gain_for_target_db:.2f} dB exceeds "
                f"maximum {self.config.max_positive_gain_db} dB - manual review recommended"
            )

        # Apply gain
        try:
            self._apply_gain(input_path, output_path, gain_calculation.applied_gain_db)
        except Exception as e:
            return FileResult(
                file=input_path.name,
                input_path=str(input_path),
                output_path=None,
                before=before_analysis,
                processing=gain_calculation,
                after=None,
                status=FileStatus.ERROR,
                error=f"Failed to apply gain: {str(e)}",
            )

        # Analyze output
        try:
            after_analysis = self.analyzer.analyze(output_path)
        except Exception as e:
            return FileResult(
                file=input_path.name,
                input_path=str(input_path),
                output_path=str(output_path),
                before=before_analysis,
                processing=gain_calculation,
                after=None,
                status=FileStatus.ERROR,
                error=f"Failed to analyze output: {str(e)}",
            )

        # Validate result
        status, additional_warnings = self.validator.validate_result(
            before_analysis, after_analysis, gain_calculation
        )
        warnings.extend(additional_warnings)

        return FileResult(
            file=input_path.name,
            input_path=str(input_path),
            output_path=str(output_path),
            before=before_analysis,
            processing=gain_calculation,
            after=after_analysis,
            status=status,
            warnings=warnings,
        )

    def _apply_gain(self, input_path: Path, output_path: Path, gain_db: float) -> None:
        """
        Apply linear gain to audio file using FFmpeg.

        Args:
            input_path: Path to input file.
            output_path: Path to output file.
            gain_db: Gain to apply in decibels.
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate linear gain
        gain_linear = 10 ** (gain_db / 20.0)

        # Use FFmpeg to apply volume filter
        # This preserves the original format and applies only linear gain
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-af", f"volume={gain_linear}",
            "-c:a", "pcm_s16le",  # Keep as 16-bit PCM WAV
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    def process_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> BatchReport:
        """
        Process all WAV files in a directory.

        Args:
            input_dir: Directory containing WAV files.
            output_dir: Directory for output files.
            progress_callback: Optional callback(current, total, result) for progress.

        Returns:
            BatchReport with all results.
        """
        # Safety check: input_dir and output_dir must not be the same
        input_resolved = input_dir.resolve()
        output_resolved = output_dir.resolve()

        if input_resolved == output_resolved:
            raise ValueError(
                f"Input and output directories are the same: {input_dir}\n"
                "This would risk overwriting source files. "
                "Please specify a different output directory."
            )

        # Check if output_dir is inside input_dir
        try:
            output_resolved.relative_to(input_resolved)
            raise ValueError(
                f"Output directory {output_dir} is inside input directory {input_dir}.\n"
                "This could lead to processing previously normalized files. "
                "Please specify an output directory outside the input directory."
            )
        except ValueError as e:
            if "is inside" in str(e):
                raise

        # Scan for WAV files
        wav_files = self.scan_wav_files(input_dir)

        if not wav_files:
            return BatchReport(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                config=self.config.to_dict(),
                files=[],
                summary=BatchSummary(
                    total_files=0,
                    processed=0,
                    passed=0,
                    warnings=0,
                    errors=0,
                    skipped=0,
                    average_lufs_before=None,
                    average_lufs_after=None,
                    maximum_true_peak_after=None,
                ),
            )

        # Process each file
        results = []
        lufs_before = []
        lufs_after = []
        true_peak_after = []

        for i, input_path in enumerate(wav_files, 1):
            output_path = output_dir / input_path.name
            result = self.process_file(input_path, output_path)
            results.append(result)

            if result.before:
                lufs_before.append(result.before.lufs)
            if result.after:
                lufs_after.append(result.after.lufs)
                true_peak_after.append(result.after.true_peak_db)

            if progress_callback:
                progress_callback(i, len(wav_files), result)

        # Calculate summary
        summary = BatchSummary(
            total_files=len(wav_files),
            processed=sum(1 for r in results if r.status != FileStatus.SKIPPED),
            passed=sum(1 for r in results if r.status == FileStatus.PASS),
            warnings=sum(1 for r in results if r.status == FileStatus.WARNING),
            errors=sum(1 for r in results if r.status == FileStatus.ERROR),
            skipped=sum(1 for r in results if r.status == FileStatus.SKIPPED),
            average_lufs_before=np.mean(lufs_before) if lufs_before else None,
            average_lufs_after=np.mean(lufs_after) if lufs_after else None,
            maximum_true_peak_after=max(true_peak_after) if true_peak_after else None,
        )

        return BatchReport(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config=self.config.to_dict(),
            files=results,
            summary=summary,
        )


def process_directory(
    input_dir: Path,
    output_dir: Path,
    config: Config,
    progress_callback: Optional[callable] = None,
) -> BatchReport:
    """
    Convenience function to process a directory.

    Args:
        input_dir: Directory containing WAV files.
        output_dir: Directory for output files.
        config: Processing configuration.
        progress_callback: Optional callback for progress updates.

    Returns:
        BatchReport with all results.
    """
    processor = Processor(config)
    return processor.process_batch(input_dir, output_dir, progress_callback)

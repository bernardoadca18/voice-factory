"""Audio analysis module using pyloudnorm and soundfile."""

import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .models import AudioAnalysis


class TruePeakMeasurementError(Exception):
    """Raised when True Peak measurement fails."""
    pass


class Analyzer:
    """Analyzes WAV files for loudness and technical properties."""

    # Threshold for detecting DC offset
    DC_OFFSET_THRESHOLD = 0.001

    # Threshold for detecting clipping (in linear amplitude)
    CLIPPING_THRESHOLD = 0.99

    def __init__(self):
        """Initialize the analyzer with ITU-R BS.1770 meter."""
        self.meter = None

    def analyze(self, file_path: Path) -> AudioAnalysis:
        """
        Analyze a WAV file and return comprehensive metrics.

        Args:
            file_path: Path to the WAV file.

        Returns:
            AudioAnalysis with all measured properties.

        Raises:
            TruePeakMeasurementError: If True Peak measurement fails.
        """
        # Read audio file
        audio, sample_rate = sf.read(str(file_path), always_2d=True)

        # Get format info
        info = sf.info(str(file_path))

        # Handle stereo by using the meter's integrated loudness
        # pyloudnorm handles multi-channel correctly
        meter = pyln.Meter(sample_rate)

        # Measure integrated loudness (LUFS)
        lufs = meter.integrated_loudness(audio)

        # Handle edge cases where loudness cannot be measured
        if lufs == float("-inf"):
            lufs = -100.0  # Very quiet or silent

        # Sample peak from PCM data (without oversampling) - using NumPy
        # This is independent from True Peak measurement
        sample_peak_db = self._measure_sample_peak_numpy(audio)

        # Measure true peak using FFmpeg ebur128 with high precision
        # True peak uses 4x oversampling to detect inter-sample peaks
        true_peak_db = self._measure_true_peak_ffmpeg(file_path)

        # Detect clipping in source
        clipping = self._detect_clipping(audio)

        # Measure RMS
        rms_linear = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(max(rms_linear, 1e-10))

        # Detect DC offset
        dc_offset = np.mean(audio)

        # Duration
        duration_seconds = len(audio) / sample_rate

        # Extract bit depth from subtype
        bit_depth = self._extract_bit_depth(info.subtype)

        return AudioAnalysis(
            filename=file_path.name,
            file_path=file_path,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=info.channels,
            bit_depth=bit_depth,
            lufs=lufs,
            sample_peak_db=sample_peak_db,
            true_peak_db=true_peak_db,
            clipping=clipping,
            rms_db=rms_db,
            dc_offset=dc_offset if abs(dc_offset) > self.DC_OFFSET_THRESHOLD else None,
        )

    def _measure_sample_peak_numpy(self, audio: np.ndarray) -> float:
        """
        Measure sample peak directly from PCM samples using NumPy.

        This is the true sample peak without any oversampling.
        Independent from True Peak measurement.

        Args:
            audio: Audio samples as numpy array.

        Returns:
            Sample peak in dBFS.
        """
        sample_peak_linear = np.max(np.abs(audio))
        sample_peak_db = 20 * np.log10(max(sample_peak_linear, 1e-10))
        return sample_peak_db

    def _measure_true_peak_ffmpeg(self, file_path: Path) -> float:
        """
        Measure true peak using FFmpeg's ebur128 filter with high precision.

        Uses 4x oversampling to detect inter-sample peaks that aren't visible
        in the PCM sample data.

        Args:
            file_path: Path to the WAV file.

        Returns:
            True peak in dBTP.

        Raises:
            TruePeakMeasurementError: If measurement fails.
        """
        try:
            # Use ebur128 with peak=sample+true to enable both measurements
            # True peak uses 4x oversampling to detect inter-sample peaks
            cmd = [
                "ffmpeg",
                "-i", str(file_path),
                "-filter_complex", "ebur128=peak=sample+true",
                "-f", "null",
                "-"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Check return code - only raise if it's a real error
            # FFmpeg may return warnings but still produce valid output
            if result.returncode != 0:
                # Check if the error is critical (not just warnings)
                stderr_lower = result.stderr.lower()
                if "error" in stderr_lower and "invalid" not in stderr_lower:
                    raise TruePeakMeasurementError(
                        f"FFmpeg failed with return code {result.returncode}: "
                        f"{result.stderr[:200]}"
                    )

            # Parse the summary section from stderr
            # FFmpeg ebur128 outputs the summary at the end with high precision
            true_peak_db = self._parse_true_peak_from_stderr(result.stderr)

            if true_peak_db is None:
                raise TruePeakMeasurementError(
                    f"Unable to parse True Peak from FFmpeg output. "
                    f"The ebur128 filter may not have produced the expected output format."
                )

            return true_peak_db

        except subprocess.TimeoutExpired:
            raise TruePeakMeasurementError(
                f"FFmpeg timed out while measuring True Peak for {file_path.name}. "
                f"File may be corrupted or too large."
            )
        except FileNotFoundError:
            raise TruePeakMeasurementError(
                "FFmpeg executable not found. Please install FFmpeg to measure True Peak."
            )
        except TruePeakMeasurementError:
            raise
        except Exception as e:
            raise TruePeakMeasurementError(
                f"Unexpected error measuring True Peak: {str(e)}"
            )

    def _parse_true_peak_from_stderr(self, stderr: str) -> Optional[float]:
        """
        Parse True Peak value from FFmpeg ebur128 stderr output.

        Looks for the Summary section which contains higher precision values.

        Args:
            stderr: FFmpeg stderr output.

        Returns:
            True Peak in dBTP, or None if not found.
        """
        lines = stderr.split("\n")

        in_true_peak_section = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect the True peak section in summary
            if stripped == "True peak:":
                in_true_peak_section = True
                continue

            # Parse peak values - look for "Peak:" line with dBFS unit
            if in_true_peak_section and "Peak:" in stripped and "dBFS" in stripped:
                try:
                    # Extract the numeric value
                    # Format: "Peak:         -3.0 dBFS"
                    parts = stripped.split("Peak:")
                    if len(parts) > 1:
                        peak_part = parts[1].strip()
                        # Extract number before dBFS
                        peak_str = peak_part.split("dBFS")[0].strip()
                        peak_value = float(peak_str)
                        return peak_value
                except (ValueError, IndexError):
                    pass

            # Reset section flag when we hit another section
            if stripped and stripped.endswith(":") and "peak" not in stripped.lower():
                in_true_peak_section = False

        return None

    def _detect_clipping(self, audio: np.ndarray) -> bool:
        """
        Detect if the audio has clipping.

        Clipping is detected if samples are very close to ±1.0,
        which indicates the signal was likely clipped during recording.

        Args:
            audio: Audio samples as numpy array.

        Returns:
            True if clipping is detected.
        """
        max_val = np.max(np.abs(audio))

        # Check if we have samples very close to digital maximum
        if max_val >= self.CLIPPING_THRESHOLD:
            # Also check for consecutive samples at the same level
            # which is a strong indicator of hard clipping
            flat = np.abs(np.diff(audio.flatten()))
            # If we have flat regions at max level, it's definitely clipped
            if max_val >= 0.999:
                return True
            # If we hit exactly 1.0 or -1.0, it's clipped
            if np.any(audio == 1.0) or np.any(audio == -1.0):
                return True

        return False

    def _extract_bit_depth(self, subtype: str) -> int:
        """
        Extract bit depth from soundfile subtype string.

        Args:
            subtype: Soundfile subtype string (e.g., "PCM_16", "PCM_24").

        Returns:
            Bit depth as integer.
        """
        # Common subtype mappings
        subtype_map = {
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "PCM_S8": 8,
            "PCM_U8": 8,
            "FLOAT": 32,
            "DOUBLE": 64,
        }

        return subtype_map.get(subtype, 16)


def analyze_file(file_path: Path) -> AudioAnalysis:
    """
    Convenience function to analyze a single file.

    Args:
        file_path: Path to the WAV file.

    Returns:
        AudioAnalysis with all measured properties.
    """
    analyzer = Analyzer()
    return analyzer.analyze(file_path)

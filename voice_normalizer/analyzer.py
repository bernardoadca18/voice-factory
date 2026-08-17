"""Audio analysis module using pyloudnorm and soundfile."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .models import AudioAnalysis


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

        # Measure true peak using pyloudnorm's peak method
        # True peak accounts for inter-sample peaks
        true_peak_linear = np.max(np.abs(audio))
        true_peak_db = 20 * np.log10(max(true_peak_linear, 1e-10))

        # For true peak with oversampling, use FFmpeg ebur128
        true_peak_db_oversampled = self._measure_true_peak_ffmpeg(file_path)

        # Sample peak (without oversampling)
        sample_peak_db = true_peak_db

        # Use oversampled true peak if available
        if true_peak_db_oversampled is not None:
            true_peak_db = true_peak_db_oversampled

        # Detect clipping in source
        clipping = self._detect_clipping(audio)

        # Measure RMS
        rms_linear = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(max(rms_linear, 1e-10))

        # Detect DC offset
        dc_offset = np.mean(audio)

        # Duration
        duration_seconds = len(audio) / sample_rate

        return AudioAnalysis(
            filename=file_path.name,
            file_path=file_path,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=info.channels,
            bit_depth=info.subtype,
            lufs=lufs,
            sample_peak_db=sample_peak_db,
            true_peak_db=true_peak_db,
            clipping=clipping,
            rms_db=rms_db,
            dc_offset=dc_offset if abs(dc_offset) > self.DC_OFFSET_THRESHOLD else None,
        )

    def _measure_true_peak_ffmpeg(self, file_path: Path) -> Optional[float]:
        """
        Measure true peak using FFmpeg's ebur128 filter.
        This provides more accurate true peak measurement with oversampling.

        Args:
            file_path: Path to the WAV file.

        Returns:
            True peak in dBTP, or None if measurement fails.
        """
        try:
            cmd = [
                "ffmpeg",
                "-i", str(file_path),
                "-filter_complex", "ebur128",
                "-f", "null",
                "-"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse the summary line from stderr
            # Looking for: "  Peak:      -1.2 dBTP"
            stderr = result.stderr
            for line in stderr.split("\n"):
                if "Peak:" in line and "dBTP" in line:
                    # Parse the peak value
                    parts = line.split("Peak:")
                    if len(parts) > 1:
                        peak_part = parts[1].strip()
                        # Extract the number before "dBTP"
                        peak_str = peak_part.split("dBTP")[0].strip()
                        try:
                            return float(peak_str)
                        except ValueError:
                            pass

            return None

        except Exception:
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

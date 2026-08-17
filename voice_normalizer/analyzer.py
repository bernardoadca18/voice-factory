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
            subtype=info.subtype,  # Preserve original subtype
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
        in the PCM sample data. Extracts True Peak from frame metadata which
        provides linear amplitude values with 3 decimal places of precision.

        Args:
            file_path: Path to the WAV file.

        Returns:
            True peak in dBTP.

        Raises:
            TruePeakMeasurementError: If measurement fails.
        """
        try:
            # Use ffprobe with ebur128 filter to extract True Peak metadata
            # The metadata provides linear amplitude values with higher precision
            # than the human-readable summary (which only has 1 decimal place)
            cmd = [
                "ffprobe",
                "-f", "lavfi",
                "-i", f"amovie={file_path},ebur128=peak=sample+true:metadata=1",
                "-show_entries", "frame_tags=lavfi.r128.true_peak",
                "-of", "csv=p=0",
                "-v", "quiet"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Check return code
            if result.returncode != 0:
                raise TruePeakMeasurementError(
                    f"FFprobe failed with return code {result.returncode}: "
                    f"{result.stderr[:200]}"
                )

            # Parse the linear amplitude value from metadata
            true_peak_linear = self._parse_true_peak_from_metadata(result.stdout)

            if true_peak_linear is None:
                raise TruePeakMeasurementError(
                    f"Unable to parse True Peak from FFprobe output. "
                    f"The ebur128 filter may not have produced metadata."
                )

            # Convert linear amplitude to dBTP
            # FFmpeg metadata reports True Peak as linear amplitude
            if true_peak_linear <= 0:
                raise TruePeakMeasurementError(
                    f"Invalid True Peak value: {true_peak_linear}"
                )

            true_peak_db = 20 * np.log10(true_peak_linear)
            return true_peak_db

        except subprocess.TimeoutExpired:
            raise TruePeakMeasurementError(
                f"FFprobe timed out while measuring True Peak for {file_path.name}. "
                f"File may be corrupted or too large."
            )
        except FileNotFoundError:
            raise TruePeakMeasurementError(
                "FFprobe executable not found. Please install FFmpeg to measure True Peak."
            )
        except TruePeakMeasurementError:
            raise
        except Exception as e:
            raise TruePeakMeasurementError(
                f"Unexpected error measuring True Peak: {str(e)}"
            )

    def _parse_true_peak_from_metadata(self, stdout: str) -> Optional[float]:
        """
        Parse True Peak linear amplitude value from FFprobe metadata output.

        FFprobe outputs frame metadata with lavfi.r128.true_peak values.
        These values are in linear amplitude (not dB) with 3 decimal places.

        For multi-channel files, we need to find the maximum true peak
        across all channels (lavfi.r128.true_peaks_ch0, ch1, etc.).

        Args:
            stdout: FFprobe stdout output.

        Returns:
            True Peak in linear amplitude, or None if not found.
        """
        lines = stdout.strip().split("\n")

        max_true_peak = 0.0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # FFprobe with -of csv=p=0 outputs just the value directly
            # Format: "0.712"
            try:
                true_peak = float(line)
                if true_peak > max_true_peak:
                    max_true_peak = true_peak
            except ValueError:
                # Skip lines that aren't valid numbers
                continue

        # Return the maximum value found, or None if no values were parsed
        return max_true_peak if max_true_peak > 0 else None

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

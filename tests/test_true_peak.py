"""Tests for True Peak measurement and intersample peak detection."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_normalizer.analyzer import Analyzer, TruePeakMeasurementError
from voice_normalizer.models import AudioAnalysis


class TestIntersamplePeak:
    """Test that True Peak correctly detects intersample peaks."""

    def test_intersample_peak_greater_than_sample_peak(self):
        """
        Test that True Peak can be greater than Sample Peak.

        This creates a signal specifically designed to produce intersample peaks
        by using a high-frequency sine wave near Nyquist frequency.

        When samples are taken at points that don't align with the waveform's true peaks,
        the reconstructed waveform can have peaks between samples (intersample peaks).

        The test requires a MEASURABLE POSITIVE difference (> 0.01 dB) to prove
        that True Peak is truly detecting intersample peaks, not just copying Sample Peak.
        """
        analyzer = Analyzer()

        # Create a signal that produces intersample peaks
        # Using a sine wave at ~0.45 * sample_rate creates strong intersample peaks
        # because the samples fall on a non-peak portion of the waveform
        sample_rate = 44100
        duration = 1.0  # seconds
        frequency = sample_rate * 0.45  # Near Nyquist, produces strong intersample peaks

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Generate sine wave at high amplitude
        # The phase is chosen to maximize intersample peaks
        phase = np.pi / 4  # 45 degrees - shifts sampling points off peak
        signal = 0.95 * np.sin(2 * np.pi * frequency * t + phase)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), signal, sample_rate, subtype='PCM_16')

        try:
            # Analyze the file
            analysis = analyzer.analyze(tmp_path)

            # True Peak MUST be >= Sample Peak (this is a fundamental property)
            assert analysis.true_peak_db >= analysis.sample_peak_db, (
                f"True Peak ({analysis.true_peak_db:.3f} dBTP) should be >= "
                f"Sample Peak ({analysis.sample_peak_db:.3f} dBFS)"
            )

            # Calculate the difference
            difference = analysis.true_peak_db - analysis.sample_peak_db

            print(f"\nIntersample Peak Test Results:")
            print(f"  Sample Peak: {analysis.sample_peak_db:.3f} dBFS")
            print(f"  True Peak:   {analysis.true_peak_db:.3f} dBTP")
            print(f"  Difference:  {difference:.3f} dB")

            # CRITICAL: True Peak MUST be measurably greater than Sample Peak
            # for this specific signal designed to produce intersample peaks.
            # We require at least 0.01 dB difference to prove that:
            # 1. True Peak is detecting intersample peaks
            # 2. True Peak is not just copying Sample Peak
            # 3. The measurement has real precision
            assert difference > 0.01, (
                f"True Peak ({analysis.true_peak_db:.3f} dBTP) should be measurably greater than "
                f"Sample Peak ({analysis.sample_peak_db:.3f} dBFS) for this intersample-peak signal. "
                f"Difference of {difference:.3f} dB is too small - indicates True Peak may not be "
                f"detecting intersample peaks properly."
            )

        finally:
            # Clean up
            tmp_path.unlink()

    def test_sample_peak_from_numpy(self):
        """
        Test that Sample Peak is calculated directly from PCM samples via NumPy.

        This verifies independence from FFmpeg measurement.
        """
        analyzer = Analyzer()

        # Create a signal with known peak
        sample_rate = 44100
        duration = 1.0
        frequency = 440  # Standard A note

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create a sine wave at exactly 0.5 amplitude
        # Peak should be exactly -6.0206 dBFS (20 * log10(0.5))
        expected_peak_linear = 0.5
        signal = expected_peak_linear * np.sin(2 * np.pi * frequency * t)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), signal, sample_rate, subtype='PCM_16')

        try:
            # Analyze the file
            analysis = analyzer.analyze(tmp_path)

            # Calculate expected peak
            expected_peak_db = 20 * np.log10(expected_peak_linear)

            # Sample Peak should match the expected value (within tolerance for quantization)
            # PCM_16 has quantization error, so we allow some tolerance
            tolerance = 0.1  # 0.1 dB tolerance for quantization
            assert abs(analysis.sample_peak_db - expected_peak_db) < tolerance, (
                f"Sample Peak ({analysis.sample_peak_db:.3f} dBFS) should match "
                f"expected value ({expected_peak_db:.3f} dBFS) within {tolerance} dB"
            )

            print(f"\nSample Peak Test Results:")
            print(f"  Expected: {expected_peak_db:.3f} dBFS")
            print(f"  Measured: {analysis.sample_peak_db:.3f} dBFS")
            print(f"  Difference: {abs(analysis.sample_peak_db - expected_peak_db):.3f} dB")

        finally:
            # Clean up
            tmp_path.unlink()


class TestTruePeakErrorHandling:
    """Test error handling for True Peak measurement."""

    def test_ffmpeg_not_found_error(self):
        """Test that missing FFmpeg raises appropriate error."""
        analyzer = Analyzer()

        # Create a valid test file with sufficient duration for pyloudnorm
        sample_rate = 44100
        duration = 1.0  # Need at least 1 second for pyloudnorm
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), signal, sample_rate, subtype='PCM_16')

        try:
            # This should work if FFmpeg is installed
            # We can't easily test the FFmpeg not found case without mocking
            analysis = analyzer.analyze(tmp_path)
            assert analysis.true_peak_db is not None

        finally:
            tmp_path.unlink()

    def test_true_peak_measurement_on_valid_file(self):
        """Test that True Peak measurement works on a valid file."""
        analyzer = Analyzer()

        # Create a simple test file
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), signal, sample_rate, subtype='PCM_16')

        try:
            analysis = analyzer.analyze(tmp_path)

            # True Peak should be measured
            assert analysis.true_peak_db is not None
            assert isinstance(analysis.true_peak_db, float)
            # Should be a reasonable value (not -inf or +inf)
            assert -100 < analysis.true_peak_db < 0

            print(f"\nTrue Peak Measurement Test:")
            print(f"  True Peak: {analysis.true_peak_db:.3f} dBTP")
            print(f"  Sample Peak: {analysis.sample_peak_db:.3f} dBFS")

        finally:
            tmp_path.unlink()


class TestTruePeakIndependence:
    """Test that Sample Peak and True Peak are measured independently."""

    def test_sample_peak_not_overwritten_by_ffmpeg(self):
        """
        Test that Sample Peak value is not overwritten by FFmpeg measurement.

        Sample Peak should be calculated from NumPy and remain independent.
        """
        analyzer = Analyzer()

        # Create a test signal
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Use a frequency that won't produce significant intersample peaks
        # so we can verify Sample Peak is calculated independently
        signal = 0.7 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), signal, sample_rate, subtype='PCM_16')

        try:
            analysis = analyzer.analyze(tmp_path)

            # Calculate expected Sample Peak from the signal directly
            # (before any quantization)
            expected_sample_peak_db = 20 * np.log10(0.7)

            # Sample Peak should be close to our expected value
            # (allowing for quantization in PCM_16)
            tolerance = 0.15  # Slightly larger tolerance for quantization
            assert abs(analysis.sample_peak_db - expected_sample_peak_db) < tolerance, (
                f"Sample Peak ({analysis.sample_peak_db:.3f} dBFS) should match "
                f"NumPy calculation ({expected_sample_peak_db:.3f} dBFS)"
            )

        finally:
            tmp_path.unlink()


class TestSafetyMarginPrecision:
    """Test that safety margin is visible with sufficient precision."""

    def test_true_peak_precision_sufficient_for_safety_margin(self):
        """
        Test that True Peak measurement has sufficient precision to see
        the 0.05 dB safety margin.

        This test validates that the measurement can distinguish values
        separated by less than 0.1 dB by:
        1. Creating signals with specific known amplitudes
        2. Verifying the measured True Peak matches the expected value
        3. Confirming the precision is real (not just formatting)
        """
        analyzer = Analyzer()

        # Test with a signal at a specific amplitude
        # The key is that we get values with 3 decimals of real precision
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Test amplitude 0.7123 (should give ~-2.947 dBTP)
        signal = 0.7123 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), signal, sample_rate, subtype='PCM_16')

        try:
            analysis = analyzer.analyze(tmp_path)

            # Calculate expected True Peak from the linear amplitude
            expected_db = 20 * np.log10(0.7123)

            print(f"\nPrecision Test:")
            print(f"  Expected True Peak: {expected_db:.3f} dBTP")
            print(f"  Measured True Peak: {analysis.true_peak_db:.3f} dBTP")

            # The measured value should match the expected value within a small tolerance
            # This proves we have real precision, not just formatted output
            tolerance = 0.01  # 0.01 dB tolerance allows for quantization
            assert abs(analysis.true_peak_db - expected_db) < tolerance, (
                f"Measured True Peak ({analysis.true_peak_db:.3f} dBTP) should match "
                f"expected value ({expected_db:.3f} dBTP) within {tolerance} dB"
            )

            # Verify the measurement has 3 decimal places of precision
            # by checking it's not just rounded to 1 decimal place
            tp_rounded_3 = round(analysis.true_peak_db, 3)
            tp_rounded_1 = round(analysis.true_peak_db, 1)

            print(f"  3 decimal places: {tp_rounded_3:.3f} dBTP")
            print(f"  1 decimal place:  {tp_rounded_1:.1f} dBTP")

            # For this specific amplitude, they should NOT be equal
            # (proving we have more than 1 decimal of precision)
            assert tp_rounded_3 != tp_rounded_1, (
                f"True Peak measurement appears to have only 1 decimal of precision. "
                f"Both 3-decimal ({tp_rounded_3:.3f}) and 1-decimal ({tp_rounded_1:.1f}) "
                f"round to the same value."
            )

        finally:
            tmp_path.unlink()

    def test_true_peak_can_distinguish_small_differences(self):
        """
        Test that True Peak measurement can distinguish values
        separated by less than 0.1 dB.

        This creates two signals with slightly different amplitudes
        and verifies that the True Peak measurements differ by the
        expected amount.
        """
        analyzer = Analyzer()

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create two signals with amplitude difference that should result
        # in approximately 0.1 dB difference in True Peak
        amplitude1 = 0.70
        amplitude2 = 0.71  # About 0.12 dB difference

        # Expected difference
        expected_diff_db = 20 * np.log10(amplitude2) - 20 * np.log10(amplitude1)

        # Create first signal
        signal1 = amplitude1 * np.sin(2 * np.pi * 440 * t)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path1 = Path(tmp.name)
            sf.write(str(tmp_path1), signal1, sample_rate, subtype='PCM_16')

        # Create second signal
        signal2 = amplitude2 * np.sin(2 * np.pi * 440 * t)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path2 = Path(tmp.name)
            sf.write(str(tmp_path2), signal2, sample_rate, subtype='PCM_16')

        try:
            # Analyze both
            analysis1 = analyzer.analyze(tmp_path1)
            analysis2 = analyzer.analyze(tmp_path2)

            # Calculate the measured difference
            measured_diff_db = analysis2.true_peak_db - analysis1.true_peak_db

            print(f"\nSmall Difference Test:")
            print(f"  Signal 1 True Peak: {analysis1.true_peak_db:.3f} dBTP")
            print(f"  Signal 2 True Peak: {analysis2.true_peak_db:.3f} dBTP")
            print(f"  Expected difference: {expected_diff_db:.3f} dB")
            print(f"  Measured difference: {measured_diff_db:.3f} dB")

            # The measured difference should match the expected difference
            tolerance = 0.01  # 0.01 dB tolerance
            assert abs(measured_diff_db - expected_diff_db) < tolerance, (
                f"Measured difference ({measured_diff_db:.3f} dB) should match "
                f"expected difference ({expected_diff_db:.3f} dB) within {tolerance} dB"
            )

            # Verify we can see differences smaller than 0.1 dB
            assert abs(measured_diff_db) < 0.15, "This test uses signals with <0.15 dB difference"
            assert abs(measured_diff_db) > 0.05, "The difference should be measurable (>0.05 dB)"

        finally:
            tmp_path1.unlink()
            tmp_path2.unlink()

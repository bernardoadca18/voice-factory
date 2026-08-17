"""Tests for validation logic."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_normalizer.config import Config
from voice_normalizer.models import AudioAnalysis, FileStatus, GainCalculation
from voice_normalizer.validator import Validator


@pytest.fixture
def config():
    """Default configuration for tests."""
    return Config(
        target_lufs=-20.0,
        max_true_peak_db=-3.0,
        max_positive_gain_db=6.0,
        acceptable_lufs_min=-22.0,
        acceptable_lufs_max=-18.0,
        reject_source_clipping=True,
    )


@pytest.fixture
def validator(config):
    """Validator instance with default config."""
    return Validator(config)


def make_analysis(
    lufs: float,
    true_peak_db: float,
    clipping: bool = False,
    channels: int = 1,
    sample_rate: int = 44100,
    duration_seconds: float = 5.0,
) -> AudioAnalysis:
    """Create a mock AudioAnalysis for testing."""
    return AudioAnalysis(
        filename="test.wav",
        file_path=Path("/tmp/test.wav"),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=16,
        subtype="PCM_16",
        lufs=lufs,
        sample_peak_db=true_peak_db,
        true_peak_db=true_peak_db,
        clipping=clipping,
    )


def make_gain_calculation(
    applied_gain_db: float = 0.0,
    gain_for_target_db: float = 0.0,
    peak_limited: bool = False,
    gain_capped: bool = False,
) -> GainCalculation:
    """Create a mock GainCalculation for testing."""
    return GainCalculation(
        gain_for_target_db=gain_for_target_db,
        gain_allowed_by_peak_db=0.0,
        applied_gain_db=applied_gain_db,
        target_reached=not peak_limited and not gain_capped,
        peak_limited=peak_limited,
        gain_capped=gain_capped,
    )


class TestSourceValidation:
    """Test cases for source file validation."""

    def test_valid_source(self, validator):
        """Valid source should pass."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-10.0)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is True
        assert warnings == []

    def test_clipping_rejected(self, validator):
        """Clipped source should be rejected when reject_source_clipping=True."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-1.0, clipping=True)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is False
        assert "clipping" in " ".join(warnings).lower()

    def test_clipping_warning_when_not_rejected(self):
        """Clipped source should produce warning when reject_source_clipping=False."""
        config = Config(reject_source_clipping=False)
        validator = Validator(config)

        analysis = make_analysis(lufs=-20.0, true_peak_db=-1.0, clipping=True)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is True  # Not rejected, just warned
        assert any("clipping" in w.lower() for w in warnings)

    def test_unusual_channels(self, validator):
        """Unusual channel count should produce warning."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-10.0, channels=6)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is True
        assert any("channel" in w.lower() for w in warnings)

    def test_low_sample_rate(self, validator):
        """Low sample rate should produce warning."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-10.0, sample_rate=8000)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is True
        assert any("sample rate" in w.lower() for w in warnings)

    def test_high_sample_rate(self, validator):
        """High sample rate should produce warning."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-10.0, sample_rate=192000)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is True
        assert any("sample rate" in w.lower() for w in warnings)

    def test_very_short_file(self, validator):
        """Very short files should produce warning."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-10.0, duration_seconds=0.3)
        is_valid, warnings = validator.validate_source(analysis)

        assert is_valid is True
        assert any("short" in w.lower() for w in warnings)


class TestResultValidation:
    """Test cases for result validation."""

    def test_pass_result(self, validator):
        """Result within acceptable range should PASS."""
        before = make_analysis(lufs=-17.0, true_peak_db=-5.0)
        after = make_analysis(lufs=-20.0, true_peak_db=-5.0)
        gain = make_gain_calculation(applied_gain_db=-3.0)

        status, warnings = validator.validate_result(before, after, gain)

        assert status == FileStatus.PASS
        assert warnings == []

    def test_error_for_lufs_outside_range(self, validator):
        """LUFS outside acceptable range should produce ERROR."""
        before = make_analysis(lufs=-15.0, true_peak_db=-10.0)
        after = make_analysis(lufs=-16.0, true_peak_db=-10.0)  # Not close to target, but peak well within limits
        gain = make_gain_calculation(applied_gain_db=-1.0, peak_limited=True)

        status, warnings = validator.validate_result(before, after, gain)

        assert status == FileStatus.ERROR
        assert any("LUFS" in w for w in warnings)

    def test_error_for_peak_exceeded(self, validator):
        """True peak exceeding maximum should produce ERROR."""
        before = make_analysis(lufs=-20.0, true_peak_db=-1.0)
        after = make_analysis(lufs=-20.0, true_peak_db=0.0)  # Above max
        gain = make_gain_calculation()

        status, warnings = validator.validate_result(before, after, gain)

        assert status == FileStatus.ERROR
        assert any("peak" in w.lower() for w in warnings)

    def test_error_for_introduced_clipping(self, validator):
        """Clipping introduced by processing should produce ERROR."""
        before = make_analysis(lufs=-20.0, true_peak_db=-5.0, clipping=False)
        after = make_analysis(lufs=-20.0, true_peak_db=0.0, clipping=True)
        gain = make_gain_calculation()

        status, warnings = validator.validate_result(before, after, gain)

        assert status == FileStatus.ERROR
        assert any("clipping" in w.lower() for w in warnings)

    def test_warning_for_peak_limited(self, validator):
        """Peak-limited gain should produce WARNING."""
        before = make_analysis(lufs=-23.0, true_peak_db=-4.0)
        after = make_analysis(lufs=-21.0, true_peak_db=-3.0)
        gain = make_gain_calculation(peak_limited=True)

        status, warnings = validator.validate_result(before, after, gain)

        assert FileStatus.WARNING in (status, FileStatus.PASS)
        assert any("peak" in w.lower() for w in warnings)

    def test_warning_for_gain_capped(self, validator):
        """Gain-capped files should produce WARNING."""
        before = make_analysis(lufs=-28.0, true_peak_db=-15.0)
        after = make_analysis(lufs=-22.0, true_peak_db=-9.0)
        gain = make_gain_calculation(
            applied_gain_db=6.0,
            gain_for_target_db=8.0,
            gain_capped=True,
        )

        status, warnings = validator.validate_result(before, after, gain)

        assert any("gain" in w.lower() or "manual" in w.lower() for w in warnings)


class TestPathSafety:
    """Test cases for path safety checks."""

    def test_same_file_rejected(self, validator, tmp_path):
        """Input and output being the same file should be rejected."""
        input_path = tmp_path / "test.wav"
        output_path = tmp_path / "test.wav"

        is_safe, error_msg = validator.check_input_output_paths(input_path, output_path)

        assert is_safe is False
        assert "same file" in error_msg.lower() or "overwrite" in error_msg.lower()

    def test_different_directories_ok(self, validator, tmp_path):
        """Different directories should be allowed."""
        input_path = tmp_path / "input" / "test.wav"
        output_path = tmp_path / "output" / "test.wav"

        is_safe, error_msg = validator.check_input_output_paths(input_path, output_path)

        assert is_safe is True
        assert error_msg == ""

    def test_same_directory_different_names_ok(self, validator, tmp_path):
        """Same directory with different filenames should be allowed."""
        input_path = tmp_path / "test.wav"
        output_path = tmp_path / "test_normalized.wav"

        is_safe, error_msg = validator.check_input_output_paths(input_path, output_path)

        assert is_safe is True
        assert error_msg == ""

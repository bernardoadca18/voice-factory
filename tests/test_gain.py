"""Tests for gain calculation logic."""

import numpy as np
import pytest

from voice_normalizer.config import Config
from voice_normalizer.models import AudioAnalysis
from voice_normalizer.normalizer import Normalizer, calculate_normalization_gain


@pytest.fixture
def config():
    """Default configuration for tests."""
    return Config(
        target_lufs=-20.0,
        max_true_peak_db=-3.0,
        max_positive_gain_db=6.0,
    )


@pytest.fixture
def normalizer(config):
    """Normalizer instance with default config."""
    return Normalizer(config)


def make_analysis(
    lufs: float,
    true_peak_db: float,
    sample_peak_db: float = None,
) -> AudioAnalysis:
    """Create a mock AudioAnalysis for testing."""
    if sample_peak_db is None:
        sample_peak_db = true_peak_db

    return AudioAnalysis(
        filename="test.wav",
        file_path="/tmp/test.wav",
        duration_seconds=5.0,
        sample_rate=44100,
        channels=1,
        bit_depth=16,
        subtype="PCM_16",
        lufs=lufs,
        sample_peak_db=sample_peak_db,
        true_peak_db=true_peak_db,
        clipping=False,
    )


class TestGainCalculation:
    """Test cases for gain calculation."""

    def test_reduce_gain_for_loud_file(self, normalizer):
        """Files above target LUFS should have negative gain applied."""
        # Input at -17 LUFS (3 dB above target)
        analysis = make_analysis(lufs=-17.0, true_peak_db=-5.0)
        result = normalizer.calculate_gain(analysis)

        assert result.applied_gain_db == pytest.approx(-3.0, abs=0.01)
        assert result.target_reached is True
        assert result.peak_limited is False

    def test_increase_gain_for_quiet_file(self, normalizer):
        """Files below target LUFS should have positive gain applied."""
        # Input at -23 LUFS (3 dB below target)
        analysis = make_analysis(lufs=-23.0, true_peak_db=-8.0)
        result = normalizer.calculate_gain(analysis)

        assert result.applied_gain_db == pytest.approx(3.0, abs=0.01)
        assert result.target_reached is True
        assert result.peak_limited is False

    def test_peak_limits_gain(self, normalizer):
        """True peak constraint should limit gain when necessary."""
        # Input at -23 LUFS needs +3 dB
        # But true peak is -4 dBTP, so only +0.95 dB is allowed (with 0.05 safety margin)
        analysis = make_analysis(lufs=-23.0, true_peak_db=-4.0)
        result = normalizer.calculate_gain(analysis)

        # Should only apply +0.95 dB to respect peak limit (with safety margin)
        assert result.applied_gain_db == pytest.approx(0.95, abs=0.01)
        assert result.target_reached is False
        assert result.peak_limited is True

    def test_max_gain_cap(self, normalizer):
        """Gain should be capped at max_positive_gain_db."""
        # Input at -30 LUFS needs +10 dB (exceeds max of +6 dB)
        analysis = make_analysis(lufs=-30.0, true_peak_db=-15.0)
        result = normalizer.calculate_gain(analysis)

        # Should be capped at +6 dB
        assert result.applied_gain_db == pytest.approx(6.0, abs=0.01)
        assert result.target_reached is False
        assert result.gain_capped is True

    def test_negative_gain_not_limited_by_max(self, normalizer):
        """Negative gain should never be limited by max_positive_gain."""
        # Input at -10 LUFS needs -10 dB reduction
        analysis = make_analysis(lufs=-10.0, true_peak_db=0.0)
        result = normalizer.calculate_gain(analysis)

        # Should apply full -10 dB
        assert result.applied_gain_db == pytest.approx(-10.0, abs=0.01)
        assert result.target_reached is True

    def test_both_constraints_active(self, normalizer):
        """When both peak and max gain constraints apply, use the minimum."""
        # Input needs +8 dB for target
        # But peak allows +1.95 dB (with 0.05 safety margin)
        # And max gain is +6 dB
        # Result: +1.95 dB (most restrictive)
        analysis = make_analysis(lufs=-28.0, true_peak_db=-5.0)
        result = normalizer.calculate_gain(analysis)

        # Peak constraint (-5 to -3.05 = +1.95) is more restrictive than max gain (+6)
        assert result.applied_gain_db == pytest.approx(1.95, abs=0.01)
        assert result.peak_limited is True
        assert result.gain_capped is False

    def test_file_at_target(self, normalizer):
        """Files already at target should have 0 dB gain."""
        analysis = make_analysis(lufs=-20.0, true_peak_db=-10.0)
        result = normalizer.calculate_gain(analysis)

        assert result.applied_gain_db == pytest.approx(0.0, abs=0.01)
        assert result.target_reached is True


class TestGainExcessiveCheck:
    """Test cases for excessive gain detection."""

    def test_excessive_gain_detected(self, normalizer):
        """Should detect when required gain exceeds maximum."""
        analysis = make_analysis(lufs=-30.0, true_peak_db=-20.0)
        calculation = normalizer.calculate_gain(analysis)

        assert normalizer.check_gain_excessive(calculation) is True

    def test_normal_gain_not_flagged(self, normalizer):
        """Should not flag normal gain requirements."""
        analysis = make_analysis(lufs=-23.0, true_peak_db=-10.0)
        calculation = normalizer.calculate_gain(analysis)

        assert normalizer.check_gain_excessive(calculation) is False

    def test_negative_gain_not_flagged(self, normalizer):
        """Should not flag negative gain (reduction)."""
        analysis = make_analysis(lufs=-15.0, true_peak_db=0.0)
        calculation = normalizer.calculate_gain(analysis)

        assert normalizer.check_gain_excessive(calculation) is False


class TestLinearGainConversion:
    """Test cases for dB to linear gain conversion."""

    def test_zero_db(self, normalizer):
        """0 dB should convert to 1.0 linear gain."""
        linear = normalizer.calculate_linear_gain(0.0)
        assert linear == pytest.approx(1.0, abs=0.001)

    def test_positive_db(self, normalizer):
        """+6 dB should approximately double amplitude."""
        linear = normalizer.calculate_linear_gain(6.0)
        assert linear == pytest.approx(2.0, abs=0.01)

    def test_negative_db(self, normalizer):
        """-6 dB should approximately halve amplitude."""
        linear = normalizer.calculate_linear_gain(-6.0)
        assert linear == pytest.approx(0.5, abs=0.01)

    def test_symmetric(self, normalizer):
        """+X dB and -X dB should be multiplicative inverses."""
        for db in [3, 6, 10, 20]:
            positive = normalizer.calculate_linear_gain(db)
            negative = normalizer.calculate_linear_gain(-db)
            assert positive * negative == pytest.approx(1.0, abs=0.001)


class TestConvenienceFunction:
    """Test the convenience function for gain calculation."""

    def test_convenience_function(self):
        """Convenience function should produce same result as class method."""
        analysis = make_analysis(lufs=-17.0, true_peak_db=-5.0)

        result = calculate_normalization_gain(
            analysis,
            target_lufs=-20.0,
            max_true_peak_db=-3.0,
            max_positive_gain_db=6.0,
        )

        assert result.applied_gain_db == pytest.approx(-3.0, abs=0.01)
        assert result.target_reached is True

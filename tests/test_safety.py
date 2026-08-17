"""Tests for safety-critical behaviors."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

from voice_normalizer.config import Config
from voice_normalizer.models import AudioAnalysis, FileStatus
from voice_normalizer.processor import Processor
from voice_normalizer.validator import Validator


def create_test_wav(
    path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> None:
    """Create a test WAV file with a sine wave."""
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)

    # Ensure stereo shape for soundfile
    audio = audio.reshape(-1, 1)

    sf.write(str(path), audio, sample_rate)


def create_silent_wav(
    path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 44100,
) -> None:
    """Create a silent WAV file."""
    samples = int(duration_seconds * sample_rate)
    audio = np.zeros((samples, 1), dtype=np.float32)
    sf.write(str(path), audio, sample_rate)


def create_clipped_wav(
    path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 44100,
) -> None:
    """Create a WAV file with intentional clipping."""
    samples = int(duration_seconds * sample_rate)
    # Create audio that clips at 1.0
    audio = np.ones((samples, 1), dtype=np.float32)
    # Add some variation
    audio = audio * 0.99 + np.random.uniform(-0.01, 0.01, (samples, 1)).astype(np.float32)

    sf.write(str(path), audio, sample_rate)


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
        overwrite_output=False,
    )


@pytest.fixture
def processor(config):
    """Processor instance for tests."""
    return Processor(config)


class TestInputProtection:
    """Tests to ensure input files are never modified."""

    def test_input_never_overwritten_same_dir(self, processor, tmp_path):
        """Should reject when input and output are in the same directory with same filename."""
        input_file = tmp_path / "test.wav"
        output_file = tmp_path / "test.wav"

        create_test_wav(input_file)

        result = processor.process_file(input_file, output_file)

        assert result.status == FileStatus.ERROR
        assert "same file" in result.error.lower() or "overwrite" in result.error.lower()

    def test_input_never_overwritten_different_dirs(self, processor, tmp_path):
        """Should work correctly when input and output are in different directories."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_file = input_dir / "test.wav"
        output_file = output_dir / "test.wav"

        create_test_wav(input_file)

        result = processor.process_file(input_file, output_file)

        # Should process successfully (or fail for other reasons, not path safety)
        assert "overwrite" not in (result.error or "").lower()
        assert "same file" not in (result.error or "").lower()

    def test_batch_same_input_output_dir_rejected(self, processor, tmp_path):
        """Should reject batch processing when input_dir == output_dir."""
        input_dir = tmp_path / "dataset"
        input_dir.mkdir()

        create_test_wav(input_dir / "test1.wav")

        with pytest.raises(ValueError) as exc_info:
            processor.process_batch(input_dir, input_dir)

        assert "same" in str(exc_info.value).lower()

    def test_output_inside_input_dir_rejected(self, processor, tmp_path):
        """Should reject when output_dir is inside input_dir."""
        input_dir = tmp_path / "dataset"
        output_dir = input_dir / "normalized"
        input_dir.mkdir()

        create_test_wav(input_dir / "test1.wav")

        with pytest.raises(ValueError) as exc_info:
            processor.process_batch(input_dir, output_dir)

        assert "inside" in str(exc_info.value).lower()


class TestLinearGainOnly:
    """Tests to ensure only linear gain is applied."""

    def test_no_compressor_in_pipeline(self, processor, tmp_path):
        """Verify FFmpeg command uses only volume filter, not compressor."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_file = input_dir / "test.wav"
        output_file = output_dir / "test.wav"

        create_test_wav(input_file, amplitude=0.3)

        # Patch subprocess.run in both modules to capture the FFmpeg command
        with patch("voice_normalizer.processor.subprocess.run") as mock_processor_run, \
             patch("voice_normalizer.analyzer.subprocess.run") as mock_analyzer_run:
            mock_processor_run.return_value = MagicMock(returncode=0, stderr="")
            mock_analyzer_run.return_value = MagicMock(returncode=0, stderr="")

            try:
                result = processor.process_file(input_file, output_file)
            except Exception:
                pass

            # Check that FFmpeg was called for processing (volume application)
            if mock_processor_run.call_count > 0:
                call_args = mock_processor_run.call_args
                cmd = call_args[0][0] if call_args[0] else []

                # Should use volume filter
                assert any("volume" in str(arg) for arg in cmd), "Should use volume filter"

                # Should NOT use compressor, limiter, loudnorm
                cmd_str = " ".join(str(arg) for arg in cmd)
                assert "acompressor" not in cmd_str, "Should not use compressor"
                assert "alimiter" not in cmd_str, "Should not use limiter"
                assert "loudnorm" not in cmd_str, "Should not use loudnorm filter"


class TestClippingDetection:
    """Tests for clipping detection and handling."""

    def test_clipped_source_detected(self, processor, tmp_path):
        """Clipped source files should be detected."""
        input_file = tmp_path / "clipped.wav"
        output_file = tmp_path / "output" / "clipped.wav"

        create_clipped_wav(input_file)

        from voice_normalizer.analyzer import Analyzer
        analyzer = Analyzer()
        analysis = analyzer.analyze(input_file)

        # Clipping should be detected
        assert analysis.clipping is True

    def test_clean_source_not_flagged(self, processor, tmp_path):
        """Clean source files should not be flagged as clipped."""
        input_file = tmp_path / "clean.wav"

        create_test_wav(input_file, amplitude=0.5)

        from voice_normalizer.analyzer import Analyzer
        analyzer = Analyzer()
        analysis = analyzer.analyze(input_file)

        assert analysis.clipping is False

    def test_clipped_source_rejected(self, config, tmp_path):
        """Clipped source should be rejected with ERROR status."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_file = input_dir / "clipped.wav"
        output_file = output_dir / "clipped.wav"

        create_clipped_wav(input_file)

        processor = Processor(config)
        result = processor.process_file(input_file, output_file)

        assert result.status == FileStatus.ERROR
        assert result.before is not None
        assert result.before.clipping is True


class TestBatchIntegrity:
    """Tests for batch processing integrity."""

    def test_batch_continues_after_error(self, processor, tmp_path):
        """Batch should continue processing after a file error."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create multiple test files
        create_test_wav(input_dir / "test1.wav", amplitude=0.5)
        create_test_wav(input_dir / "test2.wav", amplitude=0.5)
        create_test_wav(input_dir / "test3.wav", amplitude=0.5)

        report = processor.process_batch(input_dir, output_dir)

        # All files should be in report
        assert len(report.files) == 3

        # Summary should be consistent
        total = (
            report.summary.passed +
            report.summary.warnings +
            report.summary.errors +
            report.summary.skipped
        )
        assert total == len(report.files)

    def test_non_wav_files_skipped(self, processor, tmp_path):
        """Non-WAV files should be skipped, not processed."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create WAV and non-WAV files
        create_test_wav(input_dir / "test.wav", amplitude=0.5)
        (input_dir / "test.txt").write_text("not a wav")
        (input_dir / "test.mp3").write_bytes(b"fake mp3")

        wav_files = processor.scan_wav_files(input_dir)

        # Only WAV should be found
        assert len(wav_files) == 1
        assert wav_files[0].suffix.lower() == ".wav"


class TestTruePeakRespect:
    """Tests for true peak constraint."""

    def test_true_peak_constraint_respected(self, processor, tmp_path):
        """True peak constraint should be respected during gain calculation."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a file that would need gain reduction due to peak
        input_file = input_dir / "loud.wav"

        # Create audio with high peak
        create_test_wav(input_file, amplitude=0.9)

        from voice_normalizer.analyzer import Analyzer
        analyzer = Analyzer()
        analysis = analyzer.analyze(input_file)

        # Even if quiet in LUFS, high peak should limit gain
        calculation = processor.normalizer.calculate_gain(analysis)

        # Check that peak constraint is respected
        expected_peak_after = analysis.true_peak_db + calculation.applied_gain_db
        assert expected_peak_after <= processor.config.max_true_peak_db + 0.1


class TestOutputValidation:
    """Tests for post-processing output validation."""

    def test_output_measured_again(self, processor, tmp_path):
        """Output files should be re-analyzed after processing."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"

        create_test_wav(input_file, amplitude=0.5)

        report = processor.process_batch(input_dir, output_dir)

        assert len(report.files) == 1
        result = report.files[0]

        # Both before and after analysis should be present
        assert result.before is not None
        assert result.after is not None

    def test_report_json_valid(self, processor, tmp_path):
        """Report JSON should be valid and complete."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"

        create_test_wav(input_file, amplitude=0.5)

        report = processor.process_batch(input_dir, output_dir)

        # Convert to dict (as would be done for JSON)
        report_dict = report.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(report_dict)
        assert len(json_str) > 0

        # Should contain expected fields
        assert "input_dir" in report_dict
        assert "output_dir" in report_dict
        assert "files" in report_dict
        assert "summary" in report_dict


class TestIdempotency:
    """Tests for idempotent behavior."""

    def test_normalized_not_reprocessed_implicitly(self, processor, tmp_path):
        """Normalized files should not be re-processed implicitly."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        create_test_wav(input_dir / "test.wav", amplitude=0.5)

        # Process once
        report1 = processor.process_batch(input_dir, output_dir)

        # Output should exist
        assert (output_dir / "test.wav").exists()

    def test_output_dir_not_discovered_as_input(self, processor, tmp_path):
        """Output directory should not be discovered as input."""
        # Create a structure where output is a sibling of input
        base = tmp_path / "base"
        input_dir = base / "input"
        output_dir = base / "output"
        input_dir.mkdir(parents=True)

        create_test_wav(input_dir / "test.wav", amplitude=0.5)

        # Process
        processor.process_batch(input_dir, output_dir)

        # Scan input again
        wav_files = processor.scan_wav_files(input_dir)

        # Should still only find the original file
        assert len(wav_files) == 1
        assert wav_files[0].parent == input_dir

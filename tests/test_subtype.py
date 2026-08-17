"""Tests for WAV subtype preservation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_normalizer.config import Config
from voice_normalizer.processor import Processor


@pytest.fixture
def config():
    """Default configuration for tests."""
    return Config(
        target_lufs=-20.0,
        max_true_peak_db=-3.0,
        max_positive_gain_db=6.0,
        overwrite_output=True,
    )


@pytest.fixture
def processor(config):
    """Processor instance for tests."""
    return Processor(config)


def create_test_wav_with_subtype(
    path: Path,
    subtype: str,
    duration_seconds: float = 1.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> None:
    """Create a test WAV file with specific subtype."""
    samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, samples, dtype=np.float64)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)

    # Ensure stereo shape for soundfile
    audio = audio.reshape(-1, 1)

    # Convert to appropriate dtype for the subtype
    if subtype == "PCM_16":
        audio = audio.astype(np.float32)
    elif subtype == "PCM_24":
        audio = audio.astype(np.float64)
    elif subtype == "PCM_32":
        audio = audio.astype(np.float64)
    elif subtype == "FLOAT":
        audio = audio.astype(np.float32)
    elif subtype == "DOUBLE":
        audio = audio.astype(np.float64)

    sf.write(str(path), audio, sample_rate, subtype=subtype)


class TestSubtypePreservation:
    """Test that WAV subtype is preserved during processing."""

    def test_pcm_16_preserved(self, processor, tmp_path):
        """PCM_16 subtype should be preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"
        create_test_wav_with_subtype(input_file, "PCM_16")

        # Get input subtype
        input_info = sf.info(str(input_file))
        assert input_info.subtype == "PCM_16"

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output subtype
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.subtype == "PCM_16", (
            f"Output subtype {output_info.subtype} should match input subtype PCM_16"
        )

    def test_pcm_24_preserved(self, processor, tmp_path):
        """PCM_24 subtype should be preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"
        create_test_wav_with_subtype(input_file, "PCM_24")

        # Get input subtype
        input_info = sf.info(str(input_file))
        assert input_info.subtype == "PCM_24"

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output subtype
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.subtype == "PCM_24", (
            f"Output subtype {output_info.subtype} should match input subtype PCM_24"
        )

    def test_pcm_32_preserved(self, processor, tmp_path):
        """PCM_32 subtype should be preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"
        create_test_wav_with_subtype(input_file, "PCM_32")

        # Get input subtype
        input_info = sf.info(str(input_file))
        assert input_info.subtype == "PCM_32"

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output subtype
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.subtype == "PCM_32", (
            f"Output subtype {output_info.subtype} should match input subtype PCM_32"
        )

    def test_float_preserved(self, processor, tmp_path):
        """FLOAT subtype should be preserved, not converted to PCM_32."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"
        create_test_wav_with_subtype(input_file, "FLOAT")

        # Get input subtype
        input_info = sf.info(str(input_file))
        assert input_info.subtype == "FLOAT"

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output subtype
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.subtype == "FLOAT", (
            f"Output subtype {output_info.subtype} should match input subtype FLOAT. "
            f"FLOAT should NOT be converted to PCM_32."
        )

    def test_double_preserved(self, processor, tmp_path):
        """DOUBLE subtype should be preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"
        create_test_wav_with_subtype(input_file, "DOUBLE")

        # Get input subtype
        input_info = sf.info(str(input_file))
        assert input_info.subtype == "DOUBLE"

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output subtype
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.subtype == "DOUBLE", (
            f"Output subtype {output_info.subtype} should match input subtype DOUBLE"
        )

    def test_sample_rate_preserved(self, processor, tmp_path):
        """Sample rate should be preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        input_file = input_dir / "test.wav"
        create_test_wav_with_subtype(input_file, "PCM_16", sample_rate=48000)

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output sample rate
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.samplerate == 48000, (
            f"Output sample rate {output_info.samplerate} should match input sample rate 48000"
        )

    def test_channels_preserved(self, processor, tmp_path):
        """Channel count should be preserved."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create mono file
        input_file = input_dir / "test.wav"
        samples = 44100
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, samples))
        audio = audio.reshape(-1, 1)
        sf.write(str(input_file), audio, 44100, subtype="PCM_16")

        # Process
        report = processor.process_batch(input_dir, output_dir)

        # Check output channels
        output_file = output_dir / "test.wav"
        assert output_file.exists()

        output_info = sf.info(str(output_file))
        assert output_info.channels == 1, (
            f"Output channels {output_info.channels} should match input channels 1"
        )

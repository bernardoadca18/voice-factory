# Voice Dataset Normalizer

A batch WAV normalizer for voice datasets that applies **only linear gain** to achieve consistent loudness levels. Designed for preparing voice datasets before training voice conversion or TTS models.

## Purpose

This tool solves a specific problem: you have a collection of voice recordings with inconsistent loudness levels, and you need to normalize them before using them for training. The key principle is **preservation** - the tool does not attempt to "fix" or "enhance" audio, only adjust gain to achieve consistent loudness.

**What it does:**
- Measures loudness (LUFS) and true peak for each file
- Calculates optimal linear gain to reach target loudness
- Respects true peak constraints to prevent clipping
- Applies gain using FFmpeg's volume filter (linear only)
- Validates results and generates detailed reports

**What it does NOT do:**
- Compressor/limiter (dynamic range compression)
- EQ or spectral modification
- Denoise or de-reverb
- Any processing that alters the audio character

## Installation

### Requirements

- Python 3.10 or higher
- FFmpeg (must be installed separately)

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

### Install the tool

```bash
cd /path/to/voice-factory
pip install -e .
```

Or install with dev dependencies for testing:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Analyze your dataset

See current loudness levels without making changes:

```bash
voice-normalizer analyze ./gold
```

### Process your dataset

Normalize files to consistent loudness:

```bash
voice-normalizer process ./gold --output ./normalized/gold
```

### Validate processed files

Check if processed files meet quality standards:

```bash
voice-normalizer validate ./normalized/gold
```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
normalization:
  target_lufs: -20.0
  max_true_peak_db: -3.0
  true_peak_safety_margin_db: 0.05
  max_positive_gain_db: 6.0

validation:
  acceptable_lufs_min: -22.0
  acceptable_lufs_max: -18.0
  true_peak_tolerance_db: 0.02
  reject_source_clipping: true

processing:
  overwrite_output: false
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `target_lufs` | Target integrated loudness | -20.0 |
| `max_true_peak_db` | Maximum true peak allowed | -3.0 |
| `true_peak_safety_margin_db` | Safety margin applied during processing | 0.05 |
| `max_positive_gain_db` | Maximum gain that can be applied automatically (files requiring more are refused) | 6.0 |
| `acceptable_lufs_min` | Minimum acceptable LUFS after processing (files outside range get ERROR) | -22.0 |
| `acceptable_lufs_max` | Maximum acceptable LUFS after processing (files outside range get ERROR) | -18.0 |
| `true_peak_tolerance_db` | Tolerance for true peak validation | 0.02 |
| `reject_source_clipping` | Reject files with source clipping | true |
| `overwrite_output` | Overwrite existing output files | false |

## CLI Commands

### `analyze`

Analyze WAV files without processing.

```bash
voice-normalizer analyze INPUT_DIR [OPTIONS]
```

Arguments:
- `INPUT_DIR` - Directory containing WAV files

Options:
- `--config, -c` - Path to configuration file

### `process`

Process WAV files with normalization.

```bash
voice-normalizer process INPUT_DIR --output OUTPUT_DIR [OPTIONS]
```

Arguments:
- `INPUT_DIR` - Directory containing WAV files

Options:
- `--output, -o` - Output directory for normalized files (required)
- `--config, -c` - Path to configuration file
- `--overwrite` - Overwrite existing output files

### `validate`

Validate previously processed files.

```bash
voice-normalizer validate INPUT_DIR [OPTIONS]
```

Arguments:
- `INPUT_DIR` - Directory containing WAV files to validate

Options:
- `--config, -c` - Path to configuration file

## Understanding LUFS and True Peak

### LUFS (Loudness Units Full Scale)

LUFS is a standardized measure of audio loudness that better reflects how humans perceive loudness compared to simple peak or RMS measurements. It's based on the ITU-R BS.1770 standard.

- **More negative** = quieter (e.g., -30 LUFS is quiet)
- **Less negative** = louder (e.g., -10 LUFS is loud)
- **Broadcast standard** for dialogue is typically around -24 LUFS
- **Streaming platforms** often target around -14 LUFS
- **Voice datasets** commonly use -20 to -24 LUFS

For voice datasets, -20 LUFS is a good starting point - loud enough for clear audio but with headroom for downstream processing.

### True Peak (dBTP)

True Peak measures the maximum peak level including inter-sample peaks - peaks that occur between samples that could cause clipping during playback or conversion.

- Measured in dBTP (decibels True Peak)
- **0 dBTP** = digital maximum
- **-1 dBTP** = 1 dB below digital maximum
- **-3 dBTP** = recommended for production

Why care about True Peak? If you normalize based only on LUFS, you might boost a signal until it clips. True Peak constraints prevent this.

## Gain Calculation Algorithm

The normalizer uses a simple but safe algorithm:

```
1. Calculate gain needed to reach target LUFS:
   gain_for_loudness = target_lufs - measured_lufs

2. Calculate maximum gain allowed by true peak:
   gain_allowed_by_peak = max_true_peak_db - measured_true_peak_db

3. Apply the minimum of both:
   applied_gain = min(gain_for_loudness, gain_allowed_by_peak)

4. Cap positive gain at maximum:
   if applied_gain > max_positive_gain_db:
       applied_gain = max_positive_gain_db
       flag_for_review = True
```

### Examples

**Example 1: Normal case**
```
Input: -17 LUFS, -5 dBTP
Target: -20 LUFS
Result: Apply -3 dB
Output: ~-20 LUFS, ~-8 dBTP
Status: PASS
```

**Example 2: Peak-limited**
```
Input: -23 LUFS, -4 dBTP
Target: -20 LUFS
Max TP: -3 dBTP
Result: Apply +1 dB (not +3 dB - would exceed peak limit)
Output: ~-22 LUFS, -3 dBTP
Status: WARNING (target not reached)
```

**Example 3: Excessive gain required**
```
Input: -30 LUFS
Target: -20 LUFS
Required: +10 dB
Max allowed: +6 dB
Result: File refused (requires manual review)
Status: WARNING (file not processed)
```

## Why Only Linear Gain?

### Why not use a compressor/limiter?

**Compressors and limiters alter the audio character.** They reduce dynamic range by making quiet parts louder and/or loud parts quieter. This changes:

- Natural speech dynamics
- Prosody and expression
- The "feel" of the voice
- Subtle characteristics that define voice identity

For voice cloning/conversion, preserving the original character is critical. A compressor might make the voice sound more "broadcast ready" but at the cost of the very qualities that make it unique.

**Linear gain preserves everything.** It's simply multiplying the entire signal by a constant factor. The relationship between samples remains identical; only the overall amplitude changes.

### Why not use FFmpeg's loudnorm filter?

The `loudnorm` filter applies dynamic loudness normalization. It can adjust gain dynamically over time to maintain consistent loudness. While useful for some applications, this is exactly what we want to avoid for voice datasets.

Our approach:
1. Measure loudness (using ITU-R BS.1770)
2. Calculate a single gain value
3. Apply that gain as a constant multiplier

This guarantees the output is identical to the input, just louder or quieter.

### What if I need more aggressive processing?

If your recordings have significant quality issues (noise, inconsistent distance from mic, room reverb), those should be addressed separately, ideally at the recording stage. This tool is designed for good recordings with inconsistent levels, not for rescuing bad recordings.

## Reports

Each processing run generates two reports:

### JSON Report (`report.json`)

Complete details for each file with 3 decimal places precision for peaks and gain:

```json
{
  "file": "sample_01.wav",
  "input_path": "./gold/sample_01.wav",
  "output_path": "./normalized/gold/sample_01.wav",
  "before": {
    "duration_seconds": 7.1,
    "sample_rate": 44100,
    "channels": 1,
    "bit_depth": 16,
    "lufs": -17.58,
    "sample_peak_db": -1.300,
    "true_peak_db": -1.270,
    "clipping": false
  },
  "processing": {
    "gain_for_target_db": -2.420,
    "gain_allowed_by_peak_db": -1.730,
    "applied_gain_db": -2.420,
    "target_reached": true,
    "peak_limited": false,
    "gain_capped": false
  },
  "after": {
    "lufs": -20.00,
    "sample_peak_db": -3.730,
    "true_peak_db": -3.690,
    "clipping": false
  },
  "status": "PASS",
  "warnings": []
}
```

### CSV Report (`report.csv`)

Summary view for spreadsheet analysis:

| file | status | input_lufs | output_lufs | applied_gain_db | ... |
|------|--------|------------|-------------|-----------------|-----|

## File Statuses

| Status | Meaning |
|--------|---------|
| PASS | File processed successfully and meets all criteria |
| WARNING | File processed but has issues (target not reached, gain capped, etc.) |
| ERROR | File could not be processed (corrupted, clipping, etc.) |
| SKIPPED | File was skipped (not a WAV, output exists, etc.) |

## Common Warnings

**"Target LUFS not reached due to true peak constraint"**
The file's true peak limited the gain that could be applied. The file is quieter than target but safe from clipping.

**"Required positive gain exceeds automatic safety limit"**
The file was very quiet and would have required excessive gain (+6 dB limit). The file is not processed automatically and requires manual review.

**"Source audio has clipping"**
The original file has clipped samples. This cannot be fixed with gain reduction - the waveform is already damaged.

**"LUFS outside acceptable range"**
The final loudness is outside the configured acceptable range. Files outside the acceptable_lufs_min/max range receive ERROR status (not WARNING).

## Processing Your Dataset

### Recommended Workflow

1. **Organize your dataset:**
   ```
   train_dataset/
   ├── gold/           # Best quality subset
   ├── full/           # All acceptable files
   └── normalized/
       ├── gold/
       └── full/
   ```

2. **Analyze first:**
   ```bash
   voice-normalizer analyze ./gold
   ```

3. **Process each subset:**
   ```bash
   voice-normalizer process ./gold --output ./normalized/gold
   voice-normalizer process ./full --output ./normalized/full
   ```

4. **Review reports:**
   Check `report.json` for any WARNING or ERROR files.

5. **Validate results:**
   ```bash
   voice-normalizer validate ./normalized/gold
   ```

### Example: Processing the dataset

```bash
# Navigate to dataset location
cd /mnt/c/Users/Bernardo.DESKTOP-G7D68S2/Desktop/train_dataset

# Process Gold subset
voice-normalizer process ./gold --output ./normalized/gold

# Process Full subset (uses same parameters)
voice-normalizer process ./full --output ./normalized/full
```

Both Gold and Full use the same normalization parameters, ensuring consistency across your entire dataset.

## Troubleshooting

### "FFmpeg not found"

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### "Input and output directories are the same"

You must specify a different output directory. This protects your source files.

```bash
# Wrong
voice-normalizer process ./gold --output ./gold

# Correct
voice-normalizer process ./gold --output ./normalized/gold
```

### "Output directory is inside input directory"

The output cannot be a subdirectory of input. This prevents accidental re-processing.

```bash
# Wrong
voice-normalizer process ./dataset --output ./dataset/normalized

# Correct
voice-normalizer process ./dataset/gold --output ./normalized/gold
```

### "No WAV files found"

Check that your files have the `.wav` extension (case-insensitive).

### Files showing WARNING status

Review the specific warnings in `report.json`. Common causes:
- File too quiet (gain capped)
- File had high true peak (target not reached)
- Unusual sample rate or channel count

### Files showing ERROR status

Check the `error` field in `report.json`. Common causes:
- Source clipping detected
- Corrupted file
- FFmpeg processing failure

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=voice_normalizer

# Run specific test file
pytest tests/test_gain.py
```

## Technical Details

### Dependencies

- **pyloudnorm** - ITU-R BS.1770 loudness measurement
- **soundfile** - Audio file I/O
- **numpy** - Numerical operations
- **FFmpeg** - Audio processing (external)

### Supported Formats

- WAV files (PCM)
- Any sample rate
- Mono or stereo

### Limitations

- Only WAV format is supported
- No resampling or format conversion
- No repair of damaged audio
- No noise reduction or enhancement

### Technical Implementation

**True Peak Measurement**
- Uses FFmpeg's `ebur128` filter with `peak=sample+true` for 4x oversampling
- Detects inter-sample peaks that aren't visible in PCM sample data
- Raises `TruePeakMeasurementError` on failure (no silent fallback)

**Sample Peak Measurement**
- Calculated independently via NumPy from PCM samples
- Measured without oversampling (true sample peak)
- Always available even if True Peak measurement fails

**Bit Depth Preservation**
- Output files preserve the original bit depth (PCM_16, PCM_24, PCM_32, FLOAT, DOUBLE)
- No automatic conversion or bit depth reduction

**Precision**
- All peak and gain values reported with 3 decimal places
- Safety margin (0.05 dB) applied during processing to prevent edge cases
- Tolerance (0.02 dB) applied during validation for measurement uncertainty

## License

MIT License

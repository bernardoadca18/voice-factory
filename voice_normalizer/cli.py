"""CLI for the voice dataset normalizer."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .config import Config
from .models import FileStatus
from .processor import Processor
from .reports import generate_reports

app = typer.Typer(
    name="voice-normalizer",
    help="Batch WAV normalizer for voice datasets using linear gain only",
    add_completion=False,
)

console = Console()


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def load_config(config_path: Optional[Path]) -> Config:
    """Load configuration from file or use defaults."""
    if config_path and config_path.exists():
        return Config.from_yaml(config_path)
    return Config()


@app.command()
def analyze(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing WAV files to analyze",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file (default: config.yaml in current directory)",
    ),
):
    """
    Analyze WAV files without processing them.

    Reports loudness, peak levels, and technical properties for each file.
    """
    # Check for FFmpeg
    if not check_ffmpeg():
        console.print("[red]Error: FFmpeg not found. Please install FFmpeg to use this tool.[/red]")
        console.print("On Ubuntu/Debian: sudo apt install ffmpeg")
        console.print("On macOS: brew install ffmpeg")
        raise typer.Exit(1)

    # Load config
    if config_path is None:
        config_path = Path("config.yaml")
    config = load_config(config_path)

    # Create processor
    processor = Processor(config)

    # Scan for files
    console.print(f"\n[bold]Voice Dataset Normalizer[/bold]\n")
    console.print(f"Input: {input_dir}")

    try:
        wav_files = processor.scan_wav_files(input_dir)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not wav_files:
        console.print("[yellow]No WAV files found in input directory.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(wav_files)} WAV files.\n")

    # Analyze each file
    from .analyzer import Analyzer
    analyzer = Analyzer()

    table = Table(title="Analysis Results")
    table.add_column("File", style="cyan")
    table.add_column("LUFS", justify="right")
    table.add_column("TP (dBTP)", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("SR", justify="right")
    table.add_column("Ch", justify="right")
    table.add_column("Clipping", justify="center")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing files...", total=len(wav_files))

        for wav_file in wav_files:
            try:
                analysis = analyzer.analyze(wav_file)
                table.add_row(
                    analysis.filename,
                    f"{analysis.lufs:.2f}",
                    f"{analysis.true_peak_db:.2f}",
                    f"{analysis.duration_seconds:.1f}s",
                    str(analysis.sample_rate),
                    str(analysis.channels),
                    "[red]Yes[/red]" if analysis.clipping else "[green]No[/green]",
                )
            except Exception as e:
                table.add_row(
                    wav_file.name,
                    "[red]Error[/red]",
                    str(e)[:20],
                    "-",
                    "-",
                    "-",
                    "-",
                )
            progress.advance(task)

    console.print(table)

    # Summary statistics
    lufs_values = []
    tp_values = []
    clipping_count = 0

    for wav_file in wav_files:
        try:
            analysis = analyzer.analyze(wav_file)
            lufs_values.append(analysis.lufs)
            tp_values.append(analysis.true_peak_db)
            if analysis.clipping:
                clipping_count += 1
        except Exception:
            pass

    if lufs_values:
        import numpy as np
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Average LUFS: {np.mean(lufs_values):.2f}")
        console.print(f"  LUFS range: {min(lufs_values):.2f} to {max(lufs_values):.2f}")
        console.print(f"  Max True Peak: {max(tp_values):.2f} dBTP")
        if clipping_count > 0:
            console.print(f"  [red]Files with clipping: {clipping_count}[/red]")


@app.command()
def process(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing WAV files to process",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Directory for normalized output files",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file (default: config.yaml in current directory)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing output files",
    ),
):
    """
    Process WAV files with linear gain normalization.

    Analyzes, normalizes, and validates each file. Generates JSON and CSV reports.
    """
    # Check for FFmpeg
    if not check_ffmpeg():
        console.print("[red]Error: FFmpeg not found. Please install FFmpeg to use this tool.[/red]")
        console.print("On Ubuntu/Debian: sudo apt install ffmpeg")
        console.print("On macOS: brew install ffmpeg")
        raise typer.Exit(1)

    # Load config
    if config_path is None:
        config_path = Path("config.yaml")
    config = load_config(config_path)

    # Override config with CLI options
    if overwrite:
        config.overwrite_output = True

    # Resolve paths
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    # Safety check
    if input_dir == output_dir:
        console.print("[red]Error: Input and output directories cannot be the same.[/red]")
        console.print("This would risk overwriting source files.")
        raise typer.Exit(1)

    # Create processor
    processor = Processor(config)

    # Display header
    console.print(f"\n[bold]Voice Dataset Normalizer[/bold]\n")
    console.print(f"Input:  {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"\nTarget LUFS:     {config.target_lufs}")
    console.print(f"Max True Peak:   {config.max_true_peak_db} dBTP")
    console.print(f"Max Gain:        +{config.max_positive_gain_db} dB\n")

    # Scan for files
    try:
        wav_files = processor.scan_wav_files(input_dir)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not wav_files:
        console.print("[yellow]No WAV files found in input directory.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(wav_files)} WAV files.\n")

    # Process files with progress callback
    results_list = []

    def progress_callback(current: int, total: int, result):
        status_color = {
            FileStatus.PASS: "green",
            FileStatus.WARNING: "yellow",
            FileStatus.ERROR: "red",
            FileStatus.SKIPPED: "dim",
        }
        color = status_color.get(result.status, "white")
        status_text = f"[{color}]{result.status.value}[/{color}]"

        console.print(f"[{current:02d}/{total}] {result.file:<20} {status_text}")

        if result.warnings:
            for warning in result.warnings:
                console.print(f"       [dim]{warning}[/dim]")

        if result.error:
            console.print(f"       [red]Error: {result.error}[/red]")

        results_list.append(result)

    # Process batch
    try:
        report = processor.process_batch(input_dir, output_dir, progress_callback)
    except ValueError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Generate reports
    json_path, csv_path = generate_reports(report, output_dir)

    # Print summary
    console.print(f"\n{'─' * 50}\n")

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("key", style="bold")
    summary_table.add_column("value")

    summary_table.add_row("Processed:", str(report.summary.processed))
    summary_table.add_row("Passed:", f"[green]{report.summary.passed}[/green]")
    summary_table.add_row("Warnings:", f"[yellow]{report.summary.warnings}[/yellow]")
    summary_table.add_row("Errors:", f"[red]{report.summary.errors}[/red]")
    summary_table.add_row("Skipped:", f"[dim]{report.summary.skipped}[/dim]")

    console.print(summary_table)

    if report.summary.average_lufs_before is not None:
        console.print(f"\n[bold]Average loudness:[/bold]")
        console.print(f"  Before: {report.summary.average_lufs_before:.2f} LUFS")
        console.print(f"  After:  {report.summary.average_lufs_after:.2f} LUFS")

    if report.summary.maximum_true_peak_after is not None:
        console.print(f"\n[bold]Maximum True Peak:[/bold]")
        console.print(f"  {report.summary.maximum_true_peak_after:.2f} dBTP")

    console.print(f"\n[bold]Report:[/bold]")
    console.print(f"  {json_path}")
    console.print(f"  {csv_path}")


@app.command()
def validate(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing WAV files to validate",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file (default: config.yaml in current directory)",
    ),
):
    """
    Validate previously processed WAV files.

    Checks if files meet the target loudness and peak constraints.
    """
    # Check for FFmpeg
    if not check_ffmpeg():
        console.print("[red]Error: FFmpeg not found. Please install FFmpeg to use this tool.[/red]")
        raise typer.Exit(1)

    # Load config
    if config_path is None:
        config_path = Path("config.yaml")
    config = load_config(config_path)

    # Create processor and analyzer
    processor = Processor(config)
    from .analyzer import Analyzer
    from .validator import Validator
    analyzer = Analyzer()
    validator = Validator(config)

    # Scan for files
    console.print(f"\n[bold]Voice Dataset Normalizer - Validation[/bold]\n")
    console.print(f"Input: {input_dir}")

    try:
        wav_files = processor.scan_wav_files(input_dir)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not wav_files:
        console.print("[yellow]No WAV files found in input directory.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(wav_files)} WAV files.\n")

    # Validate each file
    table = Table(title="Validation Results")
    table.add_column("File", style="cyan")
    table.add_column("LUFS", justify="right")
    table.add_column("TP (dBTP)", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Issues")

    passed = 0
    warnings_count = 0
    errors_count = 0

    for wav_file in wav_files:
        try:
            analysis = analyzer.analyze(wav_file)

            # Use centralized validation logic
            status, warnings_list = validator.validate_normalized_file(analysis)

            # Map status to display
            status_display = {
                FileStatus.PASS: "[green]PASS[/green]",
                FileStatus.WARNING: "[yellow]WARN[/yellow]",
                FileStatus.ERROR: "[red]FAIL[/red]",
            }

            if status == FileStatus.PASS:
                passed += 1
            elif status == FileStatus.WARNING:
                warnings_count += 1
            else:
                errors_count += 1

            table.add_row(
                analysis.filename,
                f"{analysis.lufs:.2f}",
                f"{analysis.true_peak_db:.2f}",
                status_display[status],
                "; ".join(warnings_list) if warnings_list else "-",
            )

        except Exception as e:
            table.add_row(
                wav_file.name,
                "[red]Error[/red]",
                "-",
                "[red]ERROR[/red]",
                str(e)[:30],
            )
            errors_count += 1

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Passed:   [green]{passed}[/green]")
    console.print(f"  Warnings: [yellow]{warnings_count}[/yellow]")
    console.print(f"  Errors:   [red]{errors_count}[/red]")


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"voice-normalizer version {__version__}")


if __name__ == "__main__":
    app()

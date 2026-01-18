"""Run name generation for falcon experiments."""

from datetime import datetime

from coolname import generate_slug


def generate_run_name() -> str:
    """Generate a memorable run name: YYMMDD-HHMM-adjective-noun."""
    date = datetime.now().strftime("%y%m%d")
    time = datetime.now().strftime("%H%M")
    slug = generate_slug(2)
    return f"{date}-{time}-{slug}"


def generate_run_dir(base_dir: str = "outputs") -> str:
    """Generate a full run directory path."""
    return f"{base_dir}/{generate_run_name()}"

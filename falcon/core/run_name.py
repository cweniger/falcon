"""Run name generation for falcon experiments."""

import random
from datetime import datetime


# Themed for falcon/SBI
ADJECTIVES = [
    # Speed/precision
    "swift", "sharp", "keen", "quick", "agile", "rapid",
    # Strength
    "bold", "fierce", "brave", "mighty", "steady", "strong",
    # Vision/clarity
    "bright", "clear", "focused", "precise", "true", "prime",
    # Sky/nature
    "soaring", "silent", "golden", "noble", "wild", "free",
    # Science vibes
    "deep", "vast", "dense", "sparse", "latent", "prior",
]

NOUNS = [
    # Birds of prey
    "falcon", "eagle", "hawk", "talon", "wing", "feather",
    # Cosmos
    "nebula", "quasar", "pulsar", "comet", "nova", "aurora",
    # Inference/math
    "tensor", "vector", "prior", "posterior", "theta", "sigma",
    # Physics
    "photon", "flux", "field", "wave", "pulse", "beam",
    # Peaks/targets
    "apex", "zenith", "vertex", "summit", "peak", "arc",
]


def generate_run_name() -> str:
    """Generate a memorable run name: adj-noun-YYMMDD-HHMM."""
    date = datetime.now().strftime("%y%m%d")
    time = datetime.now().strftime("%H%M")
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adj}-{noun}-{date}-{time}"


def generate_run_dir(base_dir: str = "outputs") -> str:
    """Generate a full run directory path."""
    return f"{base_dir}/{generate_run_name()}"

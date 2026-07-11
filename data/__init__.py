"""Real and synthetic datasets for PhasorFlow benchmarks."""

from .physionet_eeg import (
    download_dataset,
    download_subject,
    load_subject_epochs,
    epochs_to_bandpassed,
    prepare_dataset,
    DEFAULT_SUBJECTS,
    DEFAULT_RUNS,
)

__all__ = [
    "download_dataset",
    "download_subject",
    "load_subject_epochs",
    "epochs_to_bandpassed",
    "prepare_dataset",
    "DEFAULT_SUBJECTS",
    "DEFAULT_RUNS",
]

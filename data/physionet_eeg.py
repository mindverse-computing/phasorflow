"""
PhysioNet EEG Motor Movement/Imagery (eegmmidb) loader + feature extraction.

This module downloads and prepares real motor-imagery EEG for the VPC
benchmark used in the PhasorFlow / VPC papers. It is fully self-contained:
``prepare_dataset`` will download the required EDF files on first call and
cache them locally.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
Reference: Schalk et al., "BCI2000: A General-Purpose Brain-Computer Interface
(BCI) System", IEEE TBME 51(6), 2004.

Runs 4/8/12 contain imagined left-fist (annotation T1) vs. right-fist (T2)
movement. We extract Common Spatial Pattern (CSP) log-variance features, the
standard motor-imagery representation.

Dependencies: mne, scipy, scikit-learn. Install with:
    pip install mne scikit-learn scipy

Because MNE pulls in numba, if you hit a numba caching error in a restricted
sandbox, set the environment variable NUMBA_DISABLE_JIT=1 before importing.
"""

import os
import urllib.request

import numpy as np

PHYSIONET_BASE = "https://physionet.org/files/eegmmidb/1.0.0"
DEFAULT_RUNS = (4, 8, 12)          # imagined left vs. right fist
DEFAULT_SUBJECTS = tuple(range(1, 11))


def download_subject(sid, runs=DEFAULT_RUNS, dest="eeg_data", verbose=True):
    """Download the EDF files for one subject into ``dest``. Returns the list
    of local paths. Skips files that already exist."""
    os.makedirs(dest, exist_ok=True)
    paths = []
    for r in runs:
        fn = f"S{sid:03d}R{r:02d}.edf"
        local = os.path.join(dest, fn)
        if not (os.path.exists(local) and os.path.getsize(local) > 0):
            url = f"{PHYSIONET_BASE}/S{sid:03d}/{fn}"
            if verbose:
                print(f"  downloading {fn} ...")
            urllib.request.urlretrieve(url, local)
        paths.append(local)
    return paths


def download_dataset(subjects=DEFAULT_SUBJECTS, runs=DEFAULT_RUNS,
                     dest="eeg_data", verbose=True):
    """Download all requested subjects. Returns {sid: [paths]}."""
    out = {}
    for sid in subjects:
        out[sid] = download_subject(sid, runs=runs, dest=dest, verbose=verbose)
    if verbose:
        n = sum(len(v) for v in out.values())
        print(f"dataset ready: {len(subjects)} subjects, {n} EDF files in {dest}/")
    return out


def load_subject_epochs(sid, runs=DEFAULT_RUNS, dest="eeg_data",
                        tmin=0.5, tmax=3.5):
    """Load one subject's motor-imagery epochs (mne.Epochs). Downloads the
    EDF files if they are not already present."""
    import mne
    mne.set_log_level("ERROR")
    download_subject(sid, runs=runs, dest=dest, verbose=False)
    eps = []
    for r in runs:
        fn = os.path.join(dest, f"S{sid:03d}R{r:02d}.edf")
        raw = mne.io.read_raw_edf(fn, preload=True)
        raw.rename_channels({c: c.strip(".").upper() for c in raw.ch_names})
        events, eid = mne.events_from_annotations(raw)
        want = {k: v for k, v in eid.items() if k in ("T1", "T2")}
        if not want:
            continue
        ep = mne.Epochs(raw, events, event_id=want, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
        eps.append(ep)
    if not eps:
        return None
    return mne.concatenate_epochs(eps)


def epochs_to_bandpassed(ep, l_freq=8.0, h_freq=30.0):
    """Return (X, y): band-pass filtered epoch data (n_trials, n_ch, n_times)
    and binary labels (T2 vs T1). Suitable input for a CSP transformer that is
    fit inside a cross-validation fold."""
    ep_f = ep.copy().filter(l_freq, h_freq, verbose="ERROR")
    X = ep_f.get_data(picks="eeg")
    y = (ep.events[:, -1] == ep.events[:, -1].max()).astype(int)
    return X, y


def prepare_dataset(subjects=DEFAULT_SUBJECTS, runs=DEFAULT_RUNS,
                    dest="eeg_data", verbose=True):
    """One-call preparation: download (if needed) and return per-subject
    band-passed epoch arrays ready for CSP.

    Returns a list of dicts: [{'subject': sid, 'X': (n,ch,t), 'y': (n,)}].
    """
    download_dataset(subjects, runs=runs, dest=dest, verbose=verbose)
    out = []
    for sid in subjects:
        ep = load_subject_epochs(sid, runs=runs, dest=dest)
        if ep is None or len(ep) < 20:
            continue
        X, y = epochs_to_bandpassed(ep)
        if len(np.unique(y)) < 2:
            continue
        out.append({"subject": sid, "X": X, "y": y})
        if verbose:
            print(f"  S{sid}: {len(y)} trials, {X.shape[1]} channels")
    return out

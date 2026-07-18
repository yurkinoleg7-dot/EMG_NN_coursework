from __future__ import annotations

import os
from pathlib import Path

import json
import numpy as np
from scipy.io import loadmat


def get_start_end_pairs(
    out: dict[str, np.ndarray],
    step_backwards: int = 150,
    window_length: int = 2000,
    sfreq: int = 1000,
):
    """
    out: dict file exported from mat
    needed to have 'export_OnPaper', 'export_Trial'
    step_backwards: in ms, the amount of time we retreat before the first point on paper
    window_length: in ms, length of a desired window to cut our data
    sfreq: in Hz
    """

    onpaper = out["export_OnPaper"]
    trial = out["export_Trial"]

    trial = np.asarray(trial)
    onpaper = np.asarray(onpaper) == 1

    starts = np.r_[0, np.flatnonzero(trial[1:] != trial[:-1])]
    ends = np.r_[starts[1:], len(trial)]

    onpaper_idx = []
    for s, e in zip(starts, ends):
        idx = np.flatnonzero(onpaper[s:e])
        if idx.size == 0:
            continue
        else:
            onpaper_idx.append([s + int(idx[0]), s + int(idx[-1])])

    onpaper_idx = np.array(onpaper_idx)

    step_backwards = int(step_backwards / 1000 * sfreq)
    window_length = int(window_length / 1000 * sfreq)

    start_end_idx = np.array(
        [onpaper_idx[:, 0] - step_backwards, onpaper_idx[:, 0] + window_length]
    ).T  # type: ignore

    return start_end_idx


def process_subject_data(data_path, filename, subject_by_filename, **kwargs):

    file_path = os.path.join(data_path, filename)
    try:
        mat = loadmat(
            Path(file_path).as_posix(), squeeze_me=True, struct_as_record=False
        )
    except FileNotFoundError:
        print(f"File not found: {filename} in {data_path}")
        return None
    except NotImplementedError as e:
        raise NotImplementedError(
            f"Failed to load {file_path}. If this is a MATLAB v7.3 file, "
            "scipy.io.loadmat cannot read it; use h5py instead."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}") from e

    del mat["__header__"], mat["__version__"], mat["__globals__"]

    start_end_idx = get_start_end_pairs(mat, **kwargs)

    emg = np.array([mat["export_EMG"][s:e, :] for s, e in start_end_idx])
    x = np.array([mat["export_X"][s:e] for s, e in start_end_idx])
    y = np.array([mat["export_Y"][s:e] for s, e in start_end_idx])
    target = (
        np.array(
            [np.bincount(mat["export_Type"][s:e]).argmax() for s, e in start_end_idx]
        )
        - 1
    )

    ########
    # add this mf code because there is no data for "4" digit in 15Nov07 record and targets are mistaken
    if "15Nov07" in filename:
        target = np.where(target > 3, target + 1, target)
    ########

    subject = np.repeat(subject_by_filename[filename], len(emg))
    return {"emg": emg, "x": x, "y": y, "target": target, "subject": subject}


def build_dataset(
    metadata_json_path, recordings_root, *, step_backwards=150, window_length=2000
):
    # load metadata
    with open(metadata_json_path, "r") as f:
        meta = json.load(f)

    subject_by_filename = {r["filename"]: r["subject"] for r in meta["recordings"]}
    sfreq = meta["sampling"]["fs_hz_emg"]

    parts = {k: [] for k in ["emg", "x", "y", "target", "subject"]}

    for filename in subject_by_filename.keys():
        out = process_subject_data(
            recordings_root,
            filename,
            subject_by_filename,
            step_backwards=step_backwards,
            window_length=window_length,
            sfreq=sfreq,
        )
        if out is not None:
            print(f"file {filename} processed")
            for k in parts:
                parts[k].append(out[k])

    if not parts["emg"]:
        raise ValueError("No data produced (parts are empty)")

    return {k: np.concatenate(parts[k], axis=0) for k in parts}

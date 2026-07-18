import os
import json
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample

def get_start_end_pairs(
    out: dict[str, np.ndarray],
    step_backwards: int = 150,
    step_forwards: int = 150,  # space after lifting arm from paper
    sfreq: int = 1000,
):
    """
    Finding start and ebd of the movement by on_paper.
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
            # Take first and last meanings on_paper in trial
            onpaper_idx.append([s + int(idx[0]), s + int(idx[-1])])

    onpaper_idx = np.array(onpaper_idx)

    step_backwards_pts = int(step_backwards / 1000 * sfreq)
    step_forwards_pts = int(step_forwards / 1000 * sfreq)

    # Forming start and end for every window
    start_end_idx = np.array(
        [onpaper_idx[:, 0] - step_backwards_pts, onpaper_idx[:, 1] + step_forwards_pts]
    ).T

    return start_end_idx


def process_subject_data(data_path, filename, subject_by_filename, target_length=2500, **kwargs):
    
    file_path = os.path.join(data_path, filename)
    try:
        mat = loadmat(
            Path(file_path).as_posix(), squeeze_me=True, struct_as_record=False
        )
    except FileNotFoundError:
        print(f"File not found: {filename} in {data_path}")
        return None
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}") from e

    del mat["__header__"], mat["__version__"], mat["__globals__"]

    start_end_idx = get_start_end_pairs(mat, **kwargs)

    emg_list, x_list, y_list, target_list = [], [], [], []

    for s, e in start_end_idx:
        # Review at the end of the bouns of massive 
        s = max(0, s)
        e = min(len(mat["export_EMG"]), e)
        
        # If window is NaN (null) or negative -> skip
        if e <= s: 
            continue

        # Cut off raw data
        emg_raw = mat["export_EMG"][s:e, :]
        x_raw = mat["export_X"][s:e]
        y_raw = mat["export_Y"][s:e]
        
        # Defining target (the most frequent class in window)
        t_val = np.bincount(mat["export_Type"][s:e]).argmax() - 1
        
        # Resampling for target_length (axis=0 - interpolation by time-axis)
        emg_res = resample(emg_raw, target_length, axis=0)
        x_res = resample(x_raw, target_length, axis=0)
        y_res = resample(y_raw, target_length, axis=0)

        emg_list.append(emg_res)
        x_list.append(x_res)
        y_list.append(y_res)
        target_list.append(t_val)

    # Translate to numpy arrays
    emg = np.array(emg_list)
    x = np.array(x_list)
    y = np.array(y_list)
    target = np.array(target_list)

    ########
    # Fix for "15Nov07"
    if "15Nov07" in filename:
        target = np.where(target > 3, target + 1, target)
    ########

    subject = np.repeat(subject_by_filename[filename], len(emg))
    return {"emg": emg, "x": x, "y": y, "target": target, "subject": subject}


def build_dataset(
    metadata_json_path, recordings_root, *, step_backwards=150, step_forwards=150, target_length=2500
):
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
            target_length=target_length,
            step_backwards=step_backwards,
            step_forwards=step_forwards,
            sfreq=sfreq,
        )
        if out is not None:
            print(f"file {filename} processed")
            for k in parts:
                parts[k].append(out[k])

    if not parts["emg"]:
        raise ValueError("No data produced (parts are empty)")

    return {k: np.concatenate(parts[k], axis=0) for k in parts}
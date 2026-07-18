import numpy as np
import json
import os
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample

def apply_filters(emg, fs=1000):
    """F2: Bandpass 20-450Hz + Notch 50Hz"""
    nyq = 0.5 * fs
    # Check, frequency must not be over the  Nikewist limit
    high_val = min(450.0, nyq - 1)
    low, high = 20 / nyq, high_val / nyq
    
    b_b, a_b = butter(4, [low, high], btype='band')
    emg = filtfilt(b_b, a_b, emg, axis=0)
    
    b_n, a_n = iirnotch(50 / nyq, 30)
    emg = filtfilt(b_n, a_n, emg, axis=0)
    return emg

def apply_normalization(emg):
    """F3: Z-score by all recording"""
    return (emg - np.mean(emg, axis=0)) / (np.std(emg, axis=0) + 1e-8)

def process_subject_data(root, filename, subject_dict, use_f1, use_f2, use_f3, target_length=2500):
    # Load .mat file. Even with struct_as_record=False it can return like dict
    mat = loadmat(os.path.join(root, filename), squeeze_me=True, struct_as_record=False)
    
    # If mat is object, we turn into the dict for sameless
    if not isinstance(mat, dict):
        emg_full = getattr(mat, 'export_EMG')
        x_full   = getattr(mat, 'export_X')
        y_full   = getattr(mat, 'export_Y')
        onpaper  = np.asarray(getattr(mat, 'export_OnPaper')) == 1
        trial    = np.asarray(getattr(mat, 'export_Trial'))
        types    = getattr(mat, 'export_Type')
    else:
        emg_full = mat['export_EMG']
        x_full   = mat['export_X']
        y_full   = mat['export_Y']
        onpaper  = np.asarray(mat['export_OnPaper']) == 1
        trial    = np.asarray(mat['export_Trial'])
        types    = mat['export_Type']

    if use_f2:
        emg_full = apply_filters(emg_full)
    if use_f3:
        emg_full = apply_normalization(emg_full)
    
    starts = np.r_[0, np.flatnonzero(trial[1:] != trial[:-1]) + 1]
    ends = np.r_[starts[1:], len(trial)]
    
    step_back = 150 
    step_forward = 150 if use_f1 else 0 
    
    res_data = {k: [] for k in ["emg", "x", "y", "target"]}
    
    for s, e in zip(starts, ends):
        chunk_onpaper = onpaper[s:e]
        paper_idx = np.flatnonzero(chunk_onpaper)
        if paper_idx.size == 0: continue
        
        t_start = max(0, s + paper_idx[0] - step_back)
        t_end = min(len(trial), s + paper_idx[-1] + step_forward)
        
        e_chunk = emg_full[t_start:t_end, :]
        x_chunk = x_full[t_start:t_end]
        y_chunk = y_full[t_start:t_end]
        
        if e_chunk.shape[0] < 5: continue
        
        res_data["emg"].append(resample(e_chunk, target_length, axis=0))
        res_data["x"].append(resample(x_chunk, target_length, axis=0))
        res_data["y"].append(resample(y_chunk, target_length, axis=0))
        
        t_val = np.bincount(types[s:e].astype(int)).argmax() - 1
        if "15Nov07" in filename:
            t_val = t_val + 1 if t_val > 3 else t_val
        res_data["target"].append(t_val)
        
    if not res_data["target"]:
        return None

    return {
        "emg": np.array(res_data["emg"]), 
        "x": np.array(res_data["x"]), 
        "y": np.array(res_data["y"]), 
        "target": np.array(res_data["target"]), 
        "subject": np.repeat(subject_dict[filename], len(res_data["target"]))
    }

def build_dataset_custom(metadata_json_path, recordings_root, use_f1=False, use_f2=False, use_f3=False, target_length=2500):
    with open(metadata_json_path, "r") as f:
        meta = json.load(f)
    sub_map = {r["filename"]: r["subject"] for r in meta["recordings"]}
    
    parts = {k: [] for k in ["emg", "x", "y", "target", "subject"]}
    for fname in sub_map.keys():
        res = process_subject_data(recordings_root, fname, sub_map, use_f1, use_f2, use_f3, target_length)
        if res is not None:
            for k in parts:
                parts[k].append(res[k])
                
    if not parts["emg"]:
        raise ValueError("Данные не найдены. Проверьте пути к папке с .mat файлами.")

    return {k: np.concatenate(v, axis=0) for k in parts}

def apply_filters(emg, fs=1000):
    """
    F2 Processing: 5-450 Hz Bandpass filter + 50 Hz Notch filter.
    Eliminates motion artifacts and powerline interference.
    """
    nyq = 0.5 * fs
    low_cutoff = 5.0 
    high_cutoff = min(450.0, nyq - 1)
    low, high = low_cutoff / nyq, high_cutoff / nyq
    b_b, a_b = butter(4, [low, high], btype='band')
    emg = filtfilt(b_b, a_b, emg, axis=0)
    b_n, a_n = iirnotch(50 / nyq, 30)
    emg = filtfilt(b_n, a_n, emg, axis=0)
    return emg

def apply_normalization(emg):
    """F3 Processing: Per-channel Z-score standardization."""
    return (emg - np.mean(emg, axis=0)) / (np.std(emg, axis=0) + 1e-8)

def process_subject_data_local(root, filename, subject_dict, use_f1, use_f2, use_f3, target_length=2500):
    file_path = os.path.join(root, filename)
    try:
        mat = loadmat(Path(file_path).as_posix(), squeeze_me=True, struct_as_record=True)
        emg_full = mat['export_EMG']
        x_full   = mat['export_X']
        y_full   = mat['export_Y']
        onpaper  = np.asarray(mat['export_OnPaper']) == 1
        trial    = np.asarray(mat['export_Trial'])
        types    = mat['export_Type']
    except Exception as e:
        print(f"Skipping file {filename} due to processing error: {e}")
        return None

    if use_f2: 
        emg_full = apply_filters(emg_full)
    if use_f3: 
        emg_full = apply_normalization(emg_full)
    
    starts = np.r_[0, np.flatnonzero(trial[1:] != trial[:-1]) + 1]
    ends = np.r_[starts[1:], len(trial)]
    
    step_back = 150 
    step_forward = 150 
    res_data = {k: [] for k in ["emg", "x", "y", "target"]}
    
    for s, e in zip(starts, ends):
        if use_f1:
            # F1 Segmentation via OnPaper trace limits
            paper_idx = np.flatnonzero(onpaper[s:e])
            if paper_idx.size == 0: 
                continue
            t_start = max(0, s + paper_idx[0] - step_back)
            t_end = min(len(trial), s + paper_idx[-1] + step_forward)
        else:
            t_start, t_end = s, e
        
        e_chunk = emg_full[t_start:t_end, :]
        if e_chunk.shape[0] < 10: 
            continue
        
        res_data["emg"].append(resample(e_chunk, target_length, axis=0))
        res_data["x"].append(resample(x_full[t_start:t_end], target_length, axis=0))
        res_data["y"].append(resample(y_full[t_start:t_end], target_length, axis=0))
        
        t_val = np.bincount(types[s:e].astype(int)).argmax() - 1
        if "15Nov07" in filename:
            t_val = t_val + 1 if t_val > 3 else t_val
        res_data["target"].append(t_val)
        
    if not res_data["target"]: 
        return None
    
    return {
        "emg": np.array(res_data["emg"]), 
        "x": np.array(res_data["x"]), 
        "y": np.array(res_data["y"]), 
        "target": np.array(res_data["target"]), 
        "subject": np.repeat(subject_dict[filename], len(res_data["target"]))
    }

def build_dataset_local(metadata_json_path, recordings_root, use_f1=False, use_f2=False, use_f3=False, target_length=2500):
    with open(metadata_json_path, "r") as f:
        meta = json.load(f)
    sub_map = {r["filename"]: r["subject"] for r in meta["recordings"]}
    parts = {k: [] for k in ["emg", "x", "y", "target", "subject"]}
    for fname in sub_map.keys():
        res = process_subject_data_local(recordings_root, fname, sub_map, use_f1, use_f2, use_f3, target_length)
        if res is not None:
            print(f"Successfully processed file: {fname}")
            for k in parts:
                parts[k].append(res[k])
    if not parts["target"]:
        raise ValueError("Data pipeline execution halted: No target segments extracted.")
    return {k: np.concatenate(v_list, axis=0) for k, v_list in parts.items()}
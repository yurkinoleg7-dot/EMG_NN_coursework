# Original work info

Paper:

https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0006791

Data:

https://www.dropbox.com/sh/hgxht7wm75xkvoi/AAC1YabvD0feMdqOCWmMjQDaa?dl=0

Merged dataset for training models:

https://disk.yandex.ru/d/B5NfnwoHG4QqBg


# EMG dataset (8 channels) — description & metadata

This repository contains notebooks that build a unified `dataset.mat` from raw EMG recordings stored as MATLAB `.mat` files (e.g., `export1_20Dec07.mat`).

## What is inside the raw `.mat`

Expected keys (based on `emg_data_check.ipynb`):

- `export_EMG`: EMG signal, shape `(n_samples, 8)`
- `export_Type`: per-sample target class (digit), later converted to **0-based** labels via `export_Type - 1`
- `export_Trial`: per-sample trial id / marker used to detect trial boundaries
- `export_OnPaper`: present in the raw data, used for visualization only (not used in the final dataset build in the notebook)
- `export_X`: x coordinate of the written digit. 
- `export_Y`: y coordinate of the written digit.

## Preprocessing used to build `dataset.mat` (matches the notebook)

The notebook’s “preprocessing” is windowing + labeling (no filtering / normalization).

1. Compute trial start indices `marks` from `export_Trial`
   - Start with `marks = [0]`
   - For each index `i`, if `export_Trial[i+1] != export_Trial[i]`, append `i+1` to `marks`
2. Extract fixed-length windows around the midpoint between consecutive trial starts
   - For each pair `(marks[i], marks[i+1])` compute midpoint `round(mean(...))`
   - Extract `half_window_size` samples before and after midpoint
   - In the notebook: `half_window_size = 1400` ⇒ window length `2800` samples
3. Build labels for each window
   - Use `export_Type - 1`
   - For each target window, label = `int(mean(window))`
4. Concatenate windows from all recordings into a single array

Important notebook quirk:
- The last trial is dropped because the end index is not appended to `marks`, and the extraction loop runs over `range(len(marks) - 1)`.

## Output dataset file: `dataset.mat`

Minimal keys (as used across the notebooks):

- `data`: EMG windows, shape `(n_trials_total, 2800, 8)`
- `label`: integer label per window, shape `(n_trials_total,)`

## Recordings (subjects & dates)

Notes:

- Total: 11 recordings
- One subject (`M`) recorded on 6 different days

See `dataset_metadata.json` for the full list.

## Channel-to-muscle mapping

Provided mapping (`emg_sequence0 = [3 5 8 7 2 4 1 6]`) means:

- Channel 1 → muscle id 3 (opponens pollicis, OP)
- Channel 2 → muscle id 5 (abductor pollicis brevis, APB)
- Channel 3 → muscle id 8 (medial head of first dorsal interosseus, mFDI)
- Channel 4 → muscle id 7 (lateral head of first dorsal interosseus, lFDI)
- Channel 5 → muscle id 2 (flexor carpi radialis, FCR)
- Channel 6 → muscle id 4 (extensor digitorum, ED)
- Channel 7 → muscle id 1 (extensor carpi radialis, ECR)
- Channel 8 → muscle id 6 (extensor carpi ulnaris, ECU)

## Sampling rate

Sampling rate is 1000 Hz for EMG, 100 Hz for tablet data (resampled to 1000 Hz in the dataset).

# Classification of sEMG Signal Patterns During Handwriting Using Deep and Machine Learning

This repository contains a software suite for the recognition and classification of handwritten characters (digits 0 to 9) based on surface electromyography (sEMG) signals. The primary focus of the research is addressing the cross-subject generalization problem using hybrid neural network architectures and specialized spatial-temporal data augmentation techniques.

---

## 1. Dataset Description

The experimental data was collected according to the protocol described by *Linderman et al. (2009)*.

### Recording Parameters and Data Structure:
* **Participants:** 11 subjects. Subject `M` has 6 independent recording sessions performed on different days (`M1–M6`) to assess the temporal stability of the signals. Sessions from unique subjects (`A`, `B`, `C`, `J`) are also included.
* **Task:** Writing digits from `0` to `9` on a *Wacom Intuos 3* graphics tablet. Each digit was written approximately 50 times within a single session (totaling roughly 500 trials per session).
* **Sensors (EMG):** 8 bipolar surface electrodes recording the activity of forearm and hand muscles involved in graphomotor tasks:
  * *Forearm:* flexor carpi radialis, extensor digitorum, extensor carpi ulnaris, extensor carpi radialis.
  * *Hand:* opponens pollicis, abductor pollicis brevis, medial and lateral heads of the first dorsal interosseous muscle.
  * *Reference electrode:* placed on the forehead.
* **Technical Specifications:** Signal amplification (gain = 1000), hardware band-pass filtering (5–500 Hz), sampling rate — 1000 Hz.
* **Synchronization:** The exact temporal boundaries of the motor trials were determined using a piezoelectric film built into the tablet, which recorded direct pen-to-surface contact (`on_paper == 1`).

---

## 2. Notebooks and Repository Structure

The project is divided into research Jupyter notebooks and modular Python scripts:

* **`version-1-testing-feature.ipynb`** — A notebook dedicated to the ablation study of signal preprocessing methods. It evaluates the isolated and combined impact of window segmentation, filtering, and normalization on within-subject and cross-subject classification accuracy.
* **`baseline-compare-lr-svm-xgb-cnn-final.ipynb`** — The main analytical notebook. It contains the training pipelines for classical models (scikit-learn, XGBoost), a baseline 1D convolutional network (Simple 1D-CNN), and the final proposed architecture. It also generates comparison plots (boxplots) and the final confusion matrices.
* **`cnn_transformer.py`** / **`cnn_transformer_loso_v2.py`** — Modules implementing the final deep neural network `CNNTransformerClassifier`, custom layers, and the data loading/augmentation pipeline in PyTorch.
* **`first_model_eval.py`** — A script for the initial evaluation of baseline models and the aggregation of cross-validation logs.

---

## 3. Helper and Utility Functions

The data processing pipeline relies on the following key functions:

1. **`extract_emg_features(signal)`** — Extracts manual time-domain features for classical ML models. Calculates 4 mathematical descriptors for each of the 8 channels:
   * Mean Absolute Value (MAV)
   * Root Mean Square (RMS)
   * Variance
   * Waveform Length (WL)
2. **`set_seed(seed)`** — Ensures full experimental reproducibility by hard-fixing the random number generators for the `random`, `numpy`, and `torch` modules, as well as deterministic `torch.backends.cudnn` subsystems.
3. **`normalize_cm(cm, mode="true")`** — Normalizes confusion matrices. Supports row-wise scaling (calculating Recall for each class) or column-wise scaling (Precision).

---

## 4. Conducted Experiments

Three series of computational experiments were performed during this project:

### Experiment 1: Preprocessing Optimization (Ablation Study)
We investigated 8 possible combinations of three engineered preprocessing steps (`F1`, `F2`, `F3`) using the Leave-One-Subject-Out (LOSO) protocol on a subset of 3 contrasting subjects (`M5`, `C`, `A`):
* **`F1` (Dynamic Segmentation):** Extracting precise movement boundaries based on the `on_paper == 1` status, padded with **±150 ms** contextual margins (replacing fixed-length sliding windows).
* **`F2` (Continuous Filtering):** Applying a 4th-order Butterworth band-pass filter (20–450 Hz) and a 50 Hz notch filter to the raw continuous recording to avoid edge artifacts.
* **`F3` (Session-wide Normalization):** Z-score standardization applied per channel across the entire session of a specific user, preserving the natural amplitude proportions between different digits.

### Experiment 2: Cross-Subject Validation Scheme Optimization
Using a subset of 6 sessions from subject `M`, two Leave-One-Subject-Out (LOSO) configurations were compared:
* **Val 1:** 1 subject used for validation and Early Stopping, 1 for testing, the rest for training.
* **Val 2:** 2 subjects used for validation, 1 for testing, the rest for training.
This experiment was conducted on both raw data ("no features") and processed data ("with features").

### Experiment 3: Global Model Comparison (Within-Subject vs. LOSO)
A comparative analysis of 5 architectures was conducted using two validation protocols:
1. **Within-Subject Stratified 5-Fold CV** (the model is trained and tested on the same individual).
2. **Leave-One-Subject-Out** (the model is tested on a completely new, "unseen" user).

*Baseline models:* Multinomial Logistic Regression, Support Vector Machine (SVM with RBF kernel), Gradient Boosting (XGBoost), and a simple 1D Convolutional Neural Network (Simple 1D-CNN).
*Proposed model:* **CNN-Transformer**, utilizing a block of multi-scale parallel depthwise convolutions (kernels 21, 51, 71) to extract the EMG envelope, a projection `ChannelCNN` layer, and a `TransformerEncoder` with Multi-Head Self-Attention. It was trained using specialized spatial augmentations (simulating electrode `Bracelet shift`, `Channel dropout`, and time masking).

---

## 5. Results

### Preprocessing and Validation Results:
* **Preprocessing:** Experiment 1 demonstrated that dynamic segmentation (`F1`) is critical: without it, cross-subject LOSO accuracy drops to random guessing levels (~9.4%). Maximum synergy and stability were achieved with the full combination **`F1 + F2 + F3`** (yielding an 18.6% accuracy on the small subset without data augmentation).
* **Validation Scheme:** A statistical paired t-test revealed no significant differences between the Val 1 and Val 2 schemes (p = 0.3591). The **Val 1** scheme (1 subject for validation) was selected as the final configuration due to its lower computational complexity. Applying the `F1+F2+F3` features significantly improved classification performance across all schemes (p < 0.05).

### Final Model Comparison:

| Architecture / Model | Within-Subject Accuracy (Mean) | Cross-Subject LOSO Accuracy (Mean) |
| :--- | :---: | :---: |
| **XGBoost (Hand-crafted features)** | 0.867 | 0.254 |
| **SVM (RBF Kernel)** | 0.883 | 0.242 |
| **Logistic Regression** | 0.914 | 0.251 |
| **Simple 1D-CNN** | 0.909 | 0.186 |
| **Final CNN-Transformer** | **0.947** | **0.693** |

* **Within-Subject Scenario:** All models successfully adapted to individual motor patterns. The **CNN-Transformer** architecture achieved the highest performance (**94.7%**) while maintaining the lowest standard deviation of errors across cross-validation folds.
* **Cross-Subject (LOSO) Scenario:** Due to high inter-subject anatomical variability and electrode placement shifts, baseline models and the simple CNN failed to generalize, with accuracies dropping drastically to 18.6%–25.4%. By leveraging attention mechanisms for invariant feature extraction and robust spatial data augmentations, the proposed **CNN-Transformer** successfully overcame this barrier, achieving a mean accuracy of **69.3%** and peaking at **96.1%** for specific subjects (e.g., subject `M3`).
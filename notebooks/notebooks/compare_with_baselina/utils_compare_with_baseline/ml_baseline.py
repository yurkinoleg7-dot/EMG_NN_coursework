import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 1. REPRODUCIBILITY & UTILITIES

def set_seed(seed: int = 42):
    """
    Ensures complete reproducibility by fixing random seeds across all libraries
    and forcing deterministic execution paths in PyTorch/CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_plot_with_specs(fig, filepath: str):
    """
    Saves the matplotlib figure according to strict publication guidelines:
    PNG format, transparent background, and high density (300 DPI).
    """
    fig.savefig(filepath, format='png', transparent=True, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot successfully saved to {filepath}")


def save_experiment_metadata(params: dict, metrics_summary: dict, filepath: str):
    """
    Logs the hyperparameters, data preprocessing configuration, and final 
    aggregated model performance metrics into a structured JSON file.
    """
    output = {
        "experiment_parameters": params,
        "aggregated_metrics": metrics_summary
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Experiment metadata saved to {filepath}")

# 2. FEATURE EXTRACTION & PREPROCESSING

def extract_emg_features(X):
    """
    Extracts time-domain features from raw EMG signals for classical ML models.
    Input shape: (N, T, C) or (N, C, T)
    Output shape: (N, num_features)
    """
    if X.ndim == 3 and X.shape[1] != 8 and X.shape[2] == 8:
        X = np.transpose(X, (0, 2, 1))
        
    N, C, T = X.shape
    features_list = []
    
    for i in range(N):
        sample_features = []
        for c in range(C):
            channel_data = X[i, c, :]
            
            # Extract standard EMG time-domain features
            mav = np.mean(np.abs(channel_data))          # Mean Absolute Value
            rms = np.sqrt(np.mean(channel_data**2))      # Root Mean Square
            var = np.var(channel_data)                   # Variance
            wl = np.sum(np.abs(np.diff(channel_data)))   # Waveform Length
            
            sample_features.extend([mav, rms, var, wl])
            
        features_list.append(sample_features)
        
    return np.array(features_list)


def scale_3d_data_channels(X_train, X_val):
    """
    Performs channel-wise Z-score scaling for 3D time-series tensors (N, C, T)
    based strictly on training distribution parameters.
    """
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    for c in range(X_train.shape[1]):
        mean = X_train[:, c, :].mean()
        std = X_train[:, c, :].std() + 1e-9
        X_train_scaled[:, c, :] = (X_train[:, c, :] - mean) / std
        X_val_scaled[:, c, :] = (X_val[:, c, :] - mean) / std
    return X_train_scaled, X_val_scaled

# 3. MODEL FACTORIES

def get_logistic_regression(seed=42):
    return LogisticRegression(
        random_state=seed, 
        max_iter=2000,       
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1            
    )

def get_svm_baseline(seed=42):
    return SVC(
        kernel='rbf',
        C=1.0,           
        gamma='scale',   
        probability=True,
        random_state=seed
    )

def get_xgboost_baseline(seed=42):
    return XGBClassifier(
        objective='multi:softprob',
        num_class=10,
        random_state=seed,
        n_estimators=150,      
        max_depth=4,           
        learning_rate=0.1,     
        subsample=0.8,         
        colsample_bytree=0.8,  
        n_jobs=-1,             
        eval_metric='mlogloss' 
    )

# 4. DEEP LEARNING BENCHMARK (1D-CNN)

class SimpleCNNBaseline(nn.Module):
    """
    Standard vanilla 1D-CNN architecture acting as a baseline benchmark.
    """
    def __init__(self, in_ch=8, n_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),   
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)                                   
        )
        self.cls = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1) 
        logits = self.cls(x)
        return logits


def train_and_evaluate_simple_cnn(X_train, y_train, X_val, y_val, epochs=25, batch_size=32, lr=0.001, device='cuda', verbose=False):
    """
    Handles the training and inference pipeline for the SimpleCNNBaseline network.
    """
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    model = SimpleCNNBaseline(in_ch=8, n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            train_total += batch_X.size(0)
            
        if verbose:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, torch.LongTensor(y_val).to(device)).item()
                val_preds = val_outputs.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
            print(f"  Epoch {epoch:02d}/{epochs} | train loss: {train_loss/train_total:.4f} acc: {train_correct/train_total:.3f} | val loss: {val_loss:.4f} acc: {val_acc:.3f}")
            
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_val_t)
        preds = torch.argmax(final_outputs, dim=1).cpu().numpy()
    return preds

# 5. EVALUATION WRAPPERS WITH COMPREHENSIVE LOGGING & EXPORT

def run_ml_within_subject_universal(X_features, y_all, subject_all, model_factory, seed=42, exp_name="ml_within"):
    """
    Executes Within-Subject 5-Fold Cross-Validation for classical models.
    Logs split splits/proportions and exports quality metrics and confusion matrices to files.
    """
    set_seed(seed)
    print(f"Starting Within-Subject Validation (K=5) for {exp_name}...")
    
    unique_subjects = np.unique(subject_all)
    results = []
    global_cm = np.zeros((10, 10), dtype=int)
    total_samples = len(X_features)
    
    for subject in unique_subjects:
        subj_mask = (subject_all == subject)
        X_sub = X_features[subj_mask]
        y_sub = y_all[subj_mask]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_sub, y_sub)):
            # --- ИСПРАВЛЕНО: Теперь корректно извлекаются y_sub вместо X_sub ---
            X_train, y_train = X_sub[train_idx], y_sub[train_idx]
            X_test, y_test = X_sub[test_idx], y_sub[test_idx]
            # -----------------------------------------------------------------
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = model_factory(seed=seed)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            
            cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
            global_cm += cm
            
            results.append({
                "seed": seed,
                "fold": fold_idx + 1,
                "test_subject": subject,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_percent": round((len(X_train) / total_samples) * 100, 4),
                "test_percent": round((len(X_test) / total_samples) * 100, 4),
                "test_acc": float(acc),
                "test_f1_macro": float(f1)
            })
            print(f"Subj: {subject:3s} | Fold {fold_idx+1}/5 | Acc: {acc:.4f} | F1-Macro: {f1:.4f}")
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{exp_name}_fold_metrics.csv", index=False)
    
    summary = {
        "accuracy_mean": float(df_results["test_acc"].mean()),
        "accuracy_std": float(df_results["test_acc"].std()),
        "f1_macro_mean": float(df_results["test_f1_macro"].mean()),
        "f1_macro_std": float(df_results["test_f1_macro"].std())
    }
    
    save_experiment_metadata({"seed": seed, "validation": "within-subject", "k_folds": 5}, summary, f"{exp_name}_meta.json")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(global_cm).plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - Within Subject ({exp_name})")
    save_plot_with_specs(fig, f"{exp_name}_confusion_matrix.png")
    plt.close(fig)
    
    return results, global_cm


def run_ml_loso_universal(X_features, y_all, subject_all, model_factory, seed=42, exp_name="ml_loso"):
    """
    Executes Leave-One-Subject-Out (LOSO) Cross-Validation for classical models.
    Saves fold distribution parameters, scores (with mean/std) and saves confusion matrices.
    """
    set_seed(seed)
    subjects = np.unique(subject_all)
    results = []
    global_cm = np.zeros((10, 10), dtype=np.int64)
    total_samples = len(X_features)
    
    print(f"Starting Universal LOSO Cross-Validation across {len(subjects)} subjects...")
    
    for fold_idx, test_subject in enumerate(subjects, start=1):
        train_mask = (subject_all != test_subject)
        test_mask = (subject_all == test_subject)
        
        X_train, y_train = X_features[train_mask], y_all[train_mask]
        X_test, y_test = X_features[test_mask], y_all[test_mask]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = model_factory(seed=seed)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
        global_cm += cm
        
        results.append({
            "seed": seed,
            "fold": fold_idx,
            "test_subject": test_subject,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_percent": round(len(X_train) / total_samples * 100, 2),
            "test_percent": round(len(X_test) / total_samples * 100, 2),
            "test_acc": float(acc),
            "test_f1_macro": float(f1)
        })
        print(f"Fold {fold_idx}/{len(subjects)} | Unseen Subj: {test_subject} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{exp_name}_fold_metrics.csv", index=False)
    
    summary = {
        "accuracy_mean": float(df_results["test_acc"].mean()),
        "accuracy_std": float(df_results["test_acc"].std()),
        "f1_macro_mean": float(df_results["test_f1_macro"].mean()),
        "f1_macro_std": float(df_results["test_f1_macro"].std())
    }
    save_experiment_metadata({"seed": seed, "validation": "loso"}, summary, f"{exp_name}_meta.json")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(global_cm).plot(cmap='Oranges', ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - LOSO ({exp_name})")
    save_plot_with_specs(fig, f"{exp_name}_confusion_matrix.png")
    plt.close(fig)
    
    return results, global_cm


def run_simple_cnn_within_subject_universal_logged(X_raw, y_all, subject_all, n_splits=5, epochs=25, seed=42, exp_name="cnn_within"):
    """
    Executes Within-Subject validation for the 1D-CNN architecture.
    Saves and exports metrics, fold compositions, and the confusion matrix.
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    subjects = np.unique(subject_all)
    results = []
    global_cm = np.zeros((10, 10), dtype=np.int64)
    total_samples = len(X_raw)
    
    print(f"Starting Simple CNN Within-Subject Validation (K={n_splits}) on {device}...")
    for subj in subjects:
        subj_mask = (subject_all == subj)
        X_subj = X_raw[subj_mask]
        y_subj = y_all[subj_mask]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_subj, y_subj), start=1):
            X_train, X_val = X_subj[train_idx], X_subj[val_idx]
            y_train, y_val = y_subj[train_idx], y_subj[val_idx]
            
            X_train_s, X_val_s = scale_3d_data_channels(X_train, X_val)
            y_pred = train_and_evaluate_simple_cnn(X_train_s, y_train, X_val_s, y_val, epochs=epochs, device=device, verbose=False)
            
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")
            global_cm += confusion_matrix(y_val, y_pred, labels=list(range(10)))
            
            results.append({
                "seed": seed, "fold": fold_id, "test_subject": subj,
                "n_train": len(X_train), "n_test": len(X_val),
                "train_percent": round(len(X_train) / total_samples * 100, 4),
                "test_percent": round(len(X_val) / total_samples * 100, 4),
                "test_acc": float(acc), "test_f1_macro": float(f1)
            })
        subj_df = pd.DataFrame(results)[lambda d: d['test_subject'] == subj]
        print(f" -> Subject {subj} DONE | Mean Acc: {subj_df['test_acc'].mean():.4f} | Mean F1: {subj_df['test_f1_macro'].mean():.4f}")
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{exp_name}_fold_metrics.csv", index=False)
    
    summary = {
        "accuracy_mean": float(df_results["test_acc"].mean()),
        "accuracy_std": float(df_results["test_acc"].std()),
        "f1_macro_mean": float(df_results["test_f1_macro"].mean()),
        "f1_macro_std": float(df_results["test_f1_macro"].std())
    }
    save_experiment_metadata({"seed": seed, "architecture": "SimpleCNN", "epochs": epochs}, summary, f"{exp_name}_meta.json")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(global_cm).plot(cmap='Purples', ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - CNN Within Subject")
    save_plot_with_specs(fig, f"{exp_name}_confusion_matrix.png")
    plt.close(fig)
    
    return results, global_cm


def run_simple_cnn_loso_universal_logged(X_raw, y_all, subject_all, epochs=25, seed=42, exp_name="cnn_loso"):
    """
    Executes LOSO cross-validation for the 1D-CNN architecture.
    Provides complete pipeline logs, step-by-step reporting, and file metrics exports.
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    subjects = np.unique(subject_all)
    results = []
    global_cm = np.zeros((10, 10), dtype=np.int64)
    total_samples = len(X_raw)
    
    print(f"\nStarting Simple CNN LOSO Cross-Validation across {len(subjects)} subjects on {device}...")
    for fold_idx, test_subject in enumerate(subjects, start=1):
        print(f"\n=== Fold {fold_idx}/{len(subjects)} | Unseen Test Subject: {test_subject} ===")
        
        train_mask = (subject_all != test_subject)
        test_mask = (subject_all == test_subject)
        
        X_train, y_train = X_raw[train_mask], y_all[train_mask]
        X_test, y_test = X_raw[test_mask], y_all[test_mask]
        
        X_train_s, X_test_s = scale_3d_data_channels(X_train, X_test)
        y_pred = train_and_evaluate_simple_cnn(X_train_s, y_train, X_test_s, y_test, epochs=epochs, device=device, verbose=True)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        global_cm += confusion_matrix(y_test, y_pred, labels=list(range(10)))
        
        results.append({
            "seed": seed, "fold": fold_idx, "test_subject": test_subject,
            "n_train": len(X_train), "n_test": len(X_test),
            "train_percent": round(len(X_train) / total_samples * 100, 2),
            "test_percent": round(len(X_test) / total_samples * 100, 2),
            "test_acc": float(acc), "test_f1_macro": float(f1)
        })
        print(f"Fold {fold_idx} Result -> Subj: {test_subject} | Final Test Acc: {acc:.4f} | F1: {f1:.4f}")
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{exp_name}_fold_metrics.csv", index=False)
    
    summary = {
        "accuracy_mean": float(df_results["test_acc"].mean()),
        "accuracy_std": float(df_results["test_acc"].std()),
        "f1_macro_mean": float(df_results["test_f1_macro"].mean()),
        "f1_macro_std": float(df_results["test_f1_macro"].std())
    }
    save_experiment_metadata({"seed": seed, "architecture": "SimpleCNN", "epochs": epochs, "validation": "loso"}, summary, f"{exp_name}_meta.json")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(global_cm).plot(cmap='Reds', ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - CNN LOSO")
    save_plot_with_specs(fig, f"{exp_name}_confusion_matrix.png")
    plt.close(fig)
    
    return results, global_cm

def local_aggregate_results(results):
    """
    Локальный помощник для агрегации результатов по испытуемым и сидам,
    чтобы рассчитать итоговые метрики для сводного отчета.
    """
    df = pd.DataFrame(results)
    
    # 1. Группируем по сиду и испытуемому (усредняем фолды внутри испытуемого)
    by_test_subject = df.groupby(["seed", "test_subject"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_f1_mean=("test_f1_macro", "mean")
    ).reset_index()
    
    # 2. Группируем по сиду (усредняем по всем испытуемым)
    by_seed = by_test_subject.groupby(["seed"]).agg(
        overall_acc_mean=("test_acc_mean", "mean"),
        overall_f1_mean=("test_f1_mean", "mean")
    ).reset_index()
    
    # Возвращаем структуру, совместимую с распаковкой _, _, overall
    return None, None, {
        "acc_mean_over_seeds": float(by_seed["overall_acc_mean"].mean()),
        "f1_mean_over_seeds": float(by_seed["overall_f1_mean"].mean())
    }
import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from cnn_transformer_loso_v2_am_params import CNNTransformerClassifier

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#--------reproducibility settings--------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # детерминизм (может быть чуть медленнее)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#--------load dataset arrays-------------

def get_subject_array(dataset):
    subj = np.array(list(map(lambda x: x.replace(" ", ""), np.squeeze(dataset["subject"]))))
    return subj  # shape (N,)

def get_targets(dataset, key="target"):
    y = dataset[key]
    return np.array(y).squeeze().astype(np.int64)

def get_X(dataset, key="emg"):
    X = np.array(dataset[key])
    return X

#--------confusion matrix settings--------

def normalize_cm(cm: np.ndarray, mode: str = "true"):
    """
    mode:
      - "true": нормировка по строкам (recall per class)
      - "pred": нормировка по столбцам (precision per predicted class)
      - None: без нормировки
    """
    cm = cm.astype(np.float64)
    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    else:
        return cm
    denom = np.clip(denom, 1e-12, None)
    return cm / denom

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    normalize: str | None = "true",
    labels=None,
    figsize=(7, 6),
    savepath: str | None = None,
):
    if labels is None:
        labels = list(range(cm.shape[0]))

    cm_plot = normalize_cm(cm, normalize) if normalize else cm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # подписываем значения (аккуратно)
    # если нормировано — 2 знака, иначе целые
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            text = f"{val:.2f}" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax

def summarize_confusions(conf_mats: dict, seed=None):
    """
    conf_mats: {(seed, subject): cm_sum} как возвращается из run_within_subject_kfold
    seed: если задан, агрегируем только по нему; иначе по всем
    """
    cm_all = np.zeros((10, 10), dtype=np.int64)
    for (s, subj), cm in conf_mats.items():
        if seed is None or s == seed:
            cm_all += cm
    return cm_all

#--------reports--------

def build_reports(df: pd.DataFrame):
    """
    df: fold-level результаты (как из aggregate_results)
    возвращает: by_subj, by_seed, overall_dict
    """
    by_subj = (df.groupby(["seed", "subject"])
                 .agg(val_acc_mean=("val_acc", "mean"),
                      val_acc_std=("val_acc", "std"),
                      val_f1_mean=("val_f1_macro", "mean"),
                      val_f1_std=("val_f1_macro", "std"),
                      folds=("fold", "count"))
                 .reset_index())

    by_seed = (by_subj.groupby(["seed"])
                     .agg(subject_acc_mean=("val_acc_mean", "mean"),
                          subject_acc_std=("val_acc_mean", "std"),
                          subject_f1_mean=("val_f1_mean", "mean"),
                          subject_f1_std=("val_f1_mean", "std"))
                     .reset_index())

    overall = {
        "acc_mean_over_seeds": float(by_seed["subject_acc_mean"].mean()) if len(by_seed) else np.nan,
        "acc_std_over_seeds":  float(by_seed["subject_acc_mean"].std())  if len(by_seed) > 1 else 0.0,
        "f1_mean_over_seeds":  float(by_seed["subject_f1_mean"].mean())  if len(by_seed) else np.nan,
        "f1_std_over_seeds":   float(by_seed["subject_f1_mean"].std())   if len(by_seed) > 1 else 0.0,
    }
    return by_subj, by_seed, overall


def save_tables(df, by_subj, by_seed, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "fold_results.csv"), index=False)
    by_subj.to_csv(os.path.join(out_dir, "subject_results.csv"), index=False)
    by_seed.to_csv(os.path.join(out_dir, "seed_results.csv"), index=False)
    print(f"[OK] Saved tables to: {out_dir}/")


def print_subject_classification_reports(pred_store, seed, subjects=None):
    keys = [k for k in pred_store.keys() if k[0] == seed]
    if subjects is not None:
        keys = [k for k in keys if k[1] in set(subjects)]

    for (s, subj) in sorted(keys, key=lambda x: x[1]):
        y_true = np.concatenate(pred_store[(s, subj)]["y_true"])
        y_pred = np.concatenate(pred_store[(s, subj)]["y_pred"])
        print(f"\n=== Classification report | seed={s} subject={subj} ===")
        print(classification_report(y_true, y_pred, digits=3))

def aggregate_results(results):
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(results)
    
    # 1. Определяем, какие ключи реально присутствуют
    acc_key = 'test_acc' if 'test_acc' in df.columns else 'val_acc'
    f1_key = 'test_f1_macro' if 'test_f1_macro' in df.columns else 'val_f1_macro'
    subj_key = 'test_subject' if 'test_subject' in df.columns else 'subject'
    
    # 2. Создаем словарь для агрегации динамически
    # Это исключит ошибки, так как ключи берутся из того, что реально есть в данных
    agg_dict = {
        acc_key: ['mean', 'std'],
        f1_key: ['mean', 'std'],
        'fold': ['count']
    }
    
    # 3. Группировка
    by_subj = df.groupby(["seed", subj_key]).agg(agg_dict)
    
    # Приводим названия колонок к удобному виду (например, val_acc_mean)
    by_subj.columns = ['val_acc_mean', 'val_acc_std', 'val_f1_mean', 'val_f1_std', 'n_folds']
    by_subj = by_subj.reset_index()

    # 4. Агрегация по сидам (теперь по зафиксированным именам)
    by_seed = (by_subj.groupby(["seed"])
                      .agg(overall_acc_mean=("val_acc_mean", "mean"),
                           overall_acc_std=("val_acc_mean", "std"),
                           overall_f1_mean=("val_f1_mean", "mean"),
                           overall_f1_std=("val_f1_mean", "std"))
                      .reset_index())

    overall = {
        "acc_mean_over_seeds": float(by_seed["overall_acc_mean"].mean()),
        "acc_std_over_seeds": float(by_seed["overall_acc_mean"].std()) if len(by_seed) > 1 else 0.0,
        "f1_mean_over_seeds": float(by_seed["overall_f1_mean"].mean()),
        "f1_std_over_seeds": float(by_seed["overall_f1_mean"].std()) if len(by_seed) > 1 else 0.0
    }
    
    return by_subj, by_seed, overall

#--------torch dataset building--------
class EMGDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        norm="robust",          # "zscore" | "robust" | None
        rms_norm=True,          # дополнительно выравнивать энергию
        clip=6.0,               # None или число (например 5-8)
        eps=1e-6,
    ):
        """
        X: (N,T,C) or (N,C,T)
        y: (N,)
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

        if self.X.ndim != 3:
            raise ValueError("X must be 3D: (N,T,C) or (N,C,T)")

        # to (N,C,T)
        if self.X.shape[1] != 8 and self.X.shape[2] == 8:
            self.X = self.X.permute(0, 2, 1)

        # --- normalization over time (per trial, per channel) ---
        if norm == "zscore":
            mean = self.X.mean(dim=2, keepdim=True)
            std  = self.X.std(dim=2, keepdim=True)
            self.X = (self.X - mean) / (std + eps)

        elif norm == "robust":
            # median / MAD (MAD ~ median(|x - median|))
            med = self.X.median(dim=2, keepdim=True).values
            mad = (self.X - med).abs().median(dim=2, keepdim=True).values
            self.X = (self.X - med) / (1.4826 * mad + eps)  # 1.4826 ~ makes it comparable to std for normal dist

        elif norm is None:
            pass
        else:
            raise ValueError(f"Unknown norm={norm}")

        # --- optional: normalize energy per trial ---
        if rms_norm:
            rms = torch.sqrt((self.X ** 2).mean(dim=2, keepdim=True))
            self.X = self.X / (rms + eps)

        # --- optional: clip to reduce outliers ---
        if clip is not None:
            self.X = torch.clamp(self.X, -float(clip), float(clip))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#--------training and evaluating model--------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_y, all_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)                 # может быть Tensor или tuple
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (pred == y).sum().item()
        total += x.size(0)

        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    all_y = np.concatenate(all_y) if all_y else np.array([])
    all_pred = np.concatenate(all_pred) if all_pred else np.array([])

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
        "y_true": all_y,
        "y_pred": all_pred,
    }


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, y_main, mix_info = model(x, y)

        if mix_info is None:
            loss = nn.functional.cross_entropy(logits, y_main)
            preds = logits.argmax(dim=1)
            correct = (preds == y_main).sum().item()
        else:
            y2, lam = mix_info
            loss = (
                lam * nn.functional.cross_entropy(logits, y_main)
                + (1 - lam) * nn.functional.cross_entropy(logits, y2)
            )
            preds = logits.argmax(dim=1)
            correct = (lam * (preds == y_main).float()
                       + (1 - lam) * (preds == y2).float()).sum().item()

        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += correct
        total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def fit(model, train_loader, val_loader, epochs=30, lr=3e-4, weight_decay=1e-2,
        device=None, grad_clip=1.0, patience=17):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, grad_clip=grad_clip)
        val_out = evaluate(model, val_loader, device)
        val_loss, val_acc = val_out["loss"], val_out["acc"]

        print(f"Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best val acc: {best_val_acc:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

#------------------------------------------------------------
#--------within subject model training and evaluation--------
def subject_kfold_indices(subject_all, y_all, subject_name, n_splits=5, seed=42):
    """
    Возвращает генератор (fold_id, train_idx, val_idx) для одного subject
    """
    subj_mask = (subject_all == subject_name)
    idx = np.where(subj_mask)[0]            # глобальные индексы этого subject
    y_subj = y_all[idx]                     # метки только этого subject

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_id, (tr_local, va_local) in enumerate(skf.split(np.zeros_like(y_subj), y_subj), start=1):
        train_idx = idx[tr_local]
        val_idx   = idx[va_local]
        yield fold_id, train_idx, val_idx

def run_within_subject_kfold(
    dataset,
    subject_list=None,       # можно None => все subjects
    y_key="y",
    x_key="x",
    n_splits=5,
    base_seed=42,
    seeds=(42,),             # можно (42, 43, 44) для повторов
    batch_size=64,
    val_batch_size=128,
    epochs=500,
    lr=3e-4,
    weight_decay=1e-2,
    grad_clip=1.0,
    patience=150,
    device=None,
    model_kwargs=None,
):
    pred_store = {}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = model_kwargs or dict(in_ch=8, n_classes=10, d_model=64, stride_cnn=8, k_cnn=21)

    subject_all = get_subject_array(dataset)
    X_all = get_X(dataset, key=x_key)
    y_all = get_targets(dataset, key=y_key)

    all_subjects = np.unique(subject_all)
    if subject_list is None:
        subject_list = list(all_subjects)

    results = []
    conf_mats = {}  # (seed, subject) -> sum confusion matrix

    for seed in seeds:
        set_seed(seed)
        print(f"\n===== SEED {seed} =====")

        for subject_name in subject_list:
            print(f"\n--- Subject {subject_name} ---")
            pred_store[(seed, subject_name)] = {"y_true": [], "y_pred": []}

            # будем суммировать confusion внутри subject (по фолдам)
            cm_sum = np.zeros((10, 10), dtype=np.int64)

            for fold_id, train_idx, val_idx in subject_kfold_indices(
                subject_all, y_all, subject_name, n_splits=n_splits, seed=base_seed + seed
            ):
                print(f"\n[Subject {subject_name}] Fold {fold_id}/{n_splits} | "
                      f"train={len(train_idx)} val={len(val_idx)}")

                # datasets/loaders
                train_ds = EMGDataset(X_all[train_idx], y_all[train_idx])
                val_ds   = EMGDataset(X_all[val_idx],   y_all[val_idx])

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                          num_workers=0, pin_memory=False)
                val_loader   = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                                          num_workers=0, pin_memory=False)

                # новая модель на каждый fold
                model = CNNTransformerClassifier(**model_kwargs)

                # train
                model = fit(
                    model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                    patience=patience,
                    device=device
                )

                # финальная оценка на val
                out = evaluate(model, val_loader, device)
                y_true, y_pred = out["y_true"], out["y_pred"]
                
                pred_store[(seed, subject_name)]["y_true"].append(y_true)
                pred_store[(seed, subject_name)]["y_pred"].append(y_pred)

                fold_acc = accuracy_score(y_true, y_pred)
                fold_f1  = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
                cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
                cm_sum += cm

                results.append({
                    "seed": seed,
                    "subject": subject_name,
                    "fold": fold_id,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "val_loss": float(out["loss"]),
                    "val_acc": float(fold_acc),
                    "val_f1_macro": float(fold_f1),
                })

                print(f"[Subject {subject_name}] Fold {fold_id} DONE | "
                      f"acc={fold_acc:.3f} f1_macro={fold_f1:.3f} loss={out['loss']:.4f}")

            conf_mats[(seed, subject_name)] = cm_sum

    return results, conf_mats, pred_store


#------------------------------------------------------------
#--------all subjects model training and evaluation--------

def subject_val_test_folds(subjects_all, 
                           exclude=("S",), n_folds=5, seed=123):
    """
    Returns [(fold_id, [val_subj1, val_subj2], test_subj), ...]
    - test subjects are unique across folds (first n_folds from shuffled list)
    - val subjects chosen from remaining each fold (with RNG, can repeat across folds)
    """
    subjects_all = list(np.unique(subjects_all))
    eligible = [s for s in subjects_all if s not in set(exclude)]

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)

    if len(eligible) < 2:
        raise ValueError("Not enough eligible subjects.")

    test_subjects = eligible[:n_folds]  # unique tests
    folds = []

    for i, test_subj in enumerate(test_subjects, start=1):
        pool = [s for s in eligible if s != test_subj]
        rng.shuffle(pool)
        val_subjs = pool[:1]
        folds.append((i, val_subjs, test_subj))

    return folds


def run_train9_val2_test1_subject_kfold5(
    dataset,
    exclude_holdout_subject=("S"),
    n_folds=5,
    fold_seed=123,
    seeds=(42,),
    x_key="emg",
    y_key="target",
    batch_size=64,
    val_batch_size=128,
    test_batch_size=256,
    epochs=350,
    lr=3e-4,
    weight_decay=1e-2,
    grad_clip=1.0,
    patience=50,
    device=None,
    model_kwargs=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = model_kwargs or dict(in_ch=8, n_classes=10, d_model=64, stride_cnn=8, k_cnn=21)

    subject_all = get_subject_array(dataset)
    X_all = get_X(dataset, key=x_key)
    y_all = get_targets(dataset, key=y_key)

    folds = subject_val_test_folds(
        subject_all, exclude=exclude_holdout_subject, n_folds=n_folds, seed=fold_seed
    )

    results = []
    conf_mats = {}   # (seed, fold, split, subject) -> cm
    pred_store = {}  # (seed, fold, split, subject) -> y_true/y_pred

    for seed in seeds:
        set_seed(seed)
        print(f"\n===== SEED {seed} =====")

        for fold_id, val_subjects, test_subject in folds:
            print(f"\n=== Fold {fold_id}/{n_folds} | VAL: {val_subjects} | TEST: {test_subject} ===")

            val_idx = np.concatenate([np.where(subject_all == s)[0] for s in val_subjects])
            test_idx = np.where(subject_all == test_subject)[0]

            train_mask = np.ones(len(subject_all), dtype=bool)
            train_mask[val_idx] = False
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]

            train_loader = DataLoader(
                EMGDataset(X_all[train_idx], y_all[train_idx], norm="robust", rms_norm=True, clip=6.0),
                batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
            )
            val_loader = DataLoader(
                EMGDataset(X_all[val_idx], y_all[val_idx], norm="robust", rms_norm=True, clip=6.0),
                batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=False
            )
            test_loader = DataLoader(
                EMGDataset(X_all[test_idx], y_all[test_idx], norm="robust", rms_norm=True, clip=6.0),
                batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=False
            )

            model = CNNTransformerClassifier(**model_kwargs)

            # early stopping on combined VAL (2 subjects)
            model = fit(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                patience=patience,
                device=device
            )

            # ---- Evaluate VAL: overall (both subjects together) ----
            out_val = evaluate(model, val_loader, device)
            y_true_v, y_pred_v = out_val["y_true"], out_val["y_pred"]
            val_acc = accuracy_score(y_true_v, y_pred_v)
            val_f1  = f1_score(y_true_v, y_pred_v, average="macro", labels=np.unique(y_true_v))

            # ---- Evaluate TEST (main) ----
            out_test = evaluate(model, test_loader, device)
            y_true_t, y_pred_t = out_test["y_true"], out_test["y_pred"]
            test_acc = accuracy_score(y_true_t, y_pred_t)
            test_f1  = f1_score(y_true_t, y_pred_t, average="macro", labels=np.unique(y_true_t))

            # store per-subject conf/preds (VAL: отдельно для каждого из 2, TEST: отдельно)
            for s in val_subjects:
                idx = np.where(subject_all == s)[0]
                loader = DataLoader(EMGDataset(X_all[idx], y_all[idx], norm="robust", rms_norm=True, clip=6.0),
                                    batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=False)
                out = evaluate(model, loader, device)
                cm = confusion_matrix(out["y_true"], out["y_pred"], labels=list(range(10)))
                conf_mats[(seed, fold_id, "val", s)] = cm
                pred_store[(seed, fold_id, "val", s)] = {"y_true": [out["y_true"]], "y_pred": [out["y_pred"]]}

            cm_test = confusion_matrix(y_true_t, y_pred_t, labels=list(range(10)))
            conf_mats[(seed, fold_id, "test", test_subject)] = cm_test
            pred_store[(seed, fold_id, "test", test_subject)] = {"y_true": [y_true_t], "y_pred": [y_pred_t]}

            results.append({
                "seed": seed,
                "fold": fold_id,
                "exclude_holdout_subject": exclude_holdout_subject,
                "val_subjects": ",".join(val_subjects),
                "test_subject": test_subject,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "val_loss": float(out_val["loss"]),
                "val_acc": float(val_acc),
                "val_f1_macro": float(val_f1),
                "test_loss": float(out_test["loss"]),
                "test_acc": float(test_acc),
                "test_f1_macro": float(test_f1),
            })

            print(f"[Fold {fold_id}] VAL  {val_subjects} | acc={val_acc:.3f} f1={val_f1:.3f} loss={out_val['loss']:.4f}")
            print(f"[Fold {fold_id}] TEST {test_subject} | acc={test_acc:.3f} f1={test_f1:.3f} loss={out_test['loss']:.4f}")

            # sanity
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0

    return results, conf_mats, pred_store


def aggregate_train9_val1_test1_results(results):
    import pandas as pd
    df = pd.DataFrame(results)

    # метрики по каждому TEST subject (в пределах seed)
    by_test_subject = (df.groupby(["seed", "test_subject"])
                         .agg(test_acc_mean=("test_acc", "mean"),
                              test_acc_std=("test_acc", "std"),
                              test_f1_mean=("test_f1_macro", "mean"),
                              test_f1_std=("test_f1_macro", "std"),
                              n_folds=("fold", "count"))
                         .reset_index())

    # общий итог по seed (усреднение по тест-субъектам)
    by_seed = (by_test_subject.groupby(["seed"])
                             .agg(overall_acc_mean=("test_acc_mean", "mean"),
                                  overall_acc_std=("test_acc_mean", "std"),
                                  overall_f1_mean=("test_f1_mean", "mean"),
                                  overall_f1_std=("test_f1_mean", "std"))
                             .reset_index())

    overall = {
        "acc_mean_over_seeds": float(by_seed["overall_acc_mean"].mean()) if len(by_seed) else np.nan,
        "acc_std_over_seeds":  float(by_seed["overall_acc_mean"].std())  if len(by_seed) > 1 else 0.0,
        "f1_mean_over_seeds":  float(by_seed["overall_f1_mean"].mean())  if len(by_seed) else np.nan,
        "f1_std_over_seeds":   float(by_seed["overall_f1_mean"].std())   if len(by_seed) > 1 else 0.0,
    }

    return df, by_test_subject, by_seed, overall
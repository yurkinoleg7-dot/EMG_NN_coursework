import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from cnn_transformer import CNNTransformerClassifier

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#--------reproducibility settings--------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic (couls be slightly slower)
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
      - "true": normalization by strings (recall per class)
      - "pred": normalization by columns (precision per predicted class)
      - None: without normalization
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
    conf_mats: {(seed, subject): cm_sum} as retirns from run_within_subject_kfold
    """
    cm_all = np.zeros((10, 10), dtype=np.int64)
    for (s, subj), cm in conf_mats.items():
        if seed is None or s == seed:
            cm_all += cm
    return cm_all

#--------reports--------

def build_reports(df: pd.DataFrame):
    """
    df: fold-level results (like from aggregate_results)
    returns: by_subj, by_seed, overall_dict
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

    df = pd.DataFrame(results)

    # (a) average over folds within each subject (for each seed)
    by_subj = (df.groupby(["seed", "subject"])
                 .agg(val_acc_mean=("val_acc", "mean"),
                      val_acc_std=("val_acc", "std"),
                      val_f1_mean=("val_f1_macro", "mean"),
                      val_f1_std=("val_f1_macro", "std"))
                 .reset_index())

    # (b) average over subjects (for each seed)
    by_seed = (by_subj.groupby(["seed"])
                     .agg(subject_acc_mean=("val_acc_mean", "mean"),
                          subject_acc_std=("val_acc_mean", "std"),
                          subject_f1_mean=("val_f1_mean", "mean"),
                          subject_f1_std=("val_f1_mean", "std"))
                     .reset_index())

    # (c) overall summary across all seeds (if more than 1)
    overall = {
        "acc_mean_over_seeds": float(by_seed["subject_acc_mean"].mean()) if len(by_seed) else np.nan,
        "acc_std_over_seeds":  float(by_seed["subject_acc_mean"].std())  if len(by_seed) > 1 else 0.0,
        "f1_mean_over_seeds":  float(by_seed["subject_f1_mean"].mean())  if len(by_seed) else np.nan,
        "f1_std_over_seeds":   float(by_seed["subject_f1_mean"].std())   if len(by_seed) > 1 else 0.0,
    }

    return df, by_subj, by_seed, overall

#--------torch dataset building--------
class EMGDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy or torch, shape (N, T, C)
        y: (N,)
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

        # convert to (N, C, T)
        if self.X.ndim != 3:
            raise ValueError("X must be 3D: (N,T,C) or (N,C,T)")
        if self.X.shape[1] != 8 and self.X.shape[2] == 8:
            # (N,T,C) -> (N,C,T)
            self.X = self.X.permute(0, 2, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#--------training and evaluating model--------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    all_y = []
    all_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
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
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
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
    Returns a generator (fold_id, train_idx, val_idx) for a single subject
    """
    subj_mask = (subject_all == subject_name)
    idx = np.where(subj_mask)[0]            # global indices of this subject
    y_subj = y_all[idx]                     # labels only for this subject

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_id, (tr_local, va_local) in enumerate(skf.split(np.zeros_like(y_subj), y_subj), start=1):
        train_idx = idx[tr_local]
        val_idx   = idx[va_local]
        yield fold_id, train_idx, val_idx

def run_within_subject_kfold(
    dataset,
    subject_list=None,       # can be None => all subjects
    y_key="y",
    x_key="x",
    n_splits=5,
    base_seed=42,
    seeds=(42,),             # can be (42, 43, 44) for repeats
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

            # we will sum confusion matrix within subject (across folds)
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

                # new model for each fold
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

                # final evaluation on val
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

# def subject_val_test_folds(subjects_all, exclude_from_test=("S",), n_folds=5, seed=42):
#     """
#     Returns list of folds: [(fold_id, val_subject, test_subject), ...]
#     Uses 2*n_folds eligible subjects: for each fold takes (val, test) pair.
#     Ensures exclude_from_test subjects never appear in val/test.
#     """
#     subjects_all = list(np.unique(subjects_all))
#     eligible = [s for s in subjects_all if s not in set(exclude_from_test)]

#     rng = np.random.default_rng(seed)
#     rng.shuffle(eligible)

#     need = 2 * n_folds
#     if len(eligible) < need:
#         raise ValueError(f"Not enough eligible subjects for {n_folds} folds of (val,test). "
#                          f"Eligible={len(eligible)}, need {need}.")

#     eligible = eligible[:need]

#     folds = []
#     for i in range(n_folds):
#         val_subj = eligible[2*i]
#         test_subj = eligible[2*i + 1]
#         folds.append((i + 1, val_subj, test_subj))
#     return folds

def subject_val_test_folds(subjects_all, exclude_from_test=("S",), n_folds=5, seed=42, n_val_subjects=1):
    """
    Returns: [(fold_id, val_subject, test_subject), ...]

    val_subject:
      - if n_val_subjects==1 -> string "M2"
      - if n_val_subjects>1  -> string "M2,C" (CSV) so as not to change output structure

    - test_subj unique across folds
    - val subjects are chosen from eligible \ {test_subj} (can repeat across folds)
    - n_folds can be any value <= #eligible
    """
    subjects_all = list(np.unique(subjects_all))
    eligible = [s for s in subjects_all if s not in set(exclude_from_test)]

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)

    if n_val_subjects < 1:
       n_val_subjects=1

    if len(eligible) < (1 + n_val_subjects):
        raise ValueError(f"Not enough eligible subjects: eligible={len(eligible)}, need >= {1+n_val_subjects}")

    if n_folds > len(eligible):
        n_folds=len(eligible)

    test_subjects = eligible[:n_folds]  # unique tests
    folds = []

    for fold_id, test_subj in enumerate(test_subjects, start=1):
        pool = [s for s in eligible if s != test_subj]
        if len(pool) < n_val_subjects:
            raise ValueError(f"Not enough subjects left for val: need {n_val_subjects}, pool={len(pool)}")

        rng.shuffle(pool)
        val_subjs = pool[:n_val_subjects]

        # keep output structure: val_subject as ONE entity
        val_subject = ",".join(map(str, val_subjs))
        folds.append((fold_id, val_subject, test_subj))

    return folds


def run_train9_val1_test1_subject_kfold5(
    dataset,
    exclude_holdout_subject="S",   # this subject will not appear in val or test
    n_folds=5,
    fold_seed=123,
    n_val_subjects=2,
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
        subject_all,
        exclude_from_test=(exclude_holdout_subject,),
        n_folds=n_folds,
        seed=fold_seed,
        n_val_subjects=n_val_subjects
    )

    results = []
    conf_mats = {}   # (seed, fold, split, subject) -> cm   where split in {"val","test"}
    pred_store = {}  # (seed, fold, split, subject) -> {"y_true":[...], "y_pred":[...]}

    for seed in seeds:
        set_seed(seed)
        print(f"\n===== SEED {seed} =====")

        for fold_id, val_subject, test_subject in folds:
            print(f"\n=== Fold {fold_id}/{n_folds} | VAL: {val_subject} | TEST: {test_subject} ===")

            val_subjects = str(val_subject).split(",")  # supports both "M2" and "M2,C"
            val_idx = np.concatenate([np.where(subject_all == s)[0] for s in val_subjects])
            test_idx = np.where(subject_all == test_subject)[0]

            # train = all other subjects (9 subjects)
            train_mask = np.ones(len(subject_all), dtype=bool)
            train_mask[val_idx] = False
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]

            # loaders
            train_loader = DataLoader(
                EMGDataset(X_all[train_idx], y_all[train_idx]),
                batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
            )
            val_loader = DataLoader(
                EMGDataset(X_all[val_idx], y_all[val_idx]),
                batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=False
            )
            test_loader = DataLoader(
                EMGDataset(X_all[test_idx], y_all[test_idx]),
                batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=False
            )

            # model per fold
            model = CNNTransformerClassifier(**model_kwargs)

            # train with early stopping on VAL SUBJECT (strict)
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

            # evaluate on val subject (optional to log)
            out_val = evaluate(model, val_loader, device)
            y_true_v, y_pred_v = out_val["y_true"], out_val["y_pred"]
            val_acc = accuracy_score(y_true_v, y_pred_v)
            val_f1  = f1_score(y_true_v, y_pred_v, average="macro", labels=np.unique(y_true_v))
            cm_val = confusion_matrix(y_true_v, y_pred_v, labels=list(range(10)))

            pred_store[(seed, fold_id, "val", val_subject)] = {"y_true": [y_true_v], "y_pred": [y_pred_v]}
            conf_mats[(seed, fold_id, "val", val_subject)] = cm_val

            # evaluate on test subject (main metric)
            out_test = evaluate(model, test_loader, device)
            y_true_t, y_pred_t = out_test["y_true"], out_test["y_pred"]
            test_acc = accuracy_score(y_true_t, y_pred_t)
            test_f1  = f1_score(y_true_t, y_pred_t, average="macro", labels=np.unique(y_true_t))
            cm_test = confusion_matrix(y_true_t, y_pred_t, labels=list(range(10)))

            pred_store[(seed, fold_id, "test", test_subject)] = {"y_true": [y_true_t], "y_pred": [y_pred_t]}
            conf_mats[(seed, fold_id, "test", test_subject)] = cm_test

            results.append({
                "seed": seed,
                "fold": fold_id,
                "exclude_holdout_subject": exclude_holdout_subject,

                "val_subject": val_subject,
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

            print(f"[Fold {fold_id}] VAL  {val_subject} | acc={val_acc:.3f} f1={val_f1:.3f} loss={out_val['loss']:.4f}")
            print(f"[Fold {fold_id}] TEST {test_subject} | acc={test_acc:.3f} f1={test_f1:.3f} loss={out_test['loss']:.4f}")

            # sanity checks
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0

    return results, conf_mats, pred_store


def aggregate_train9_val1_test1_results(results):
    import pandas as pd
    df = pd.DataFrame(results)

    # metrics per TEST subject (within a seed)
    by_test_subject = (df.groupby(["seed", "test_subject"])
                         .agg(test_acc_mean=("test_acc", "mean"),
                              test_acc_std=("test_acc", "std"),
                              test_f1_mean=("test_f1_macro", "mean"),
                              test_f1_std=("test_f1_macro", "std"),
                              n_folds=("fold", "count"))
                         .reset_index())

    # overall per seed (averaged over test subjects)
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
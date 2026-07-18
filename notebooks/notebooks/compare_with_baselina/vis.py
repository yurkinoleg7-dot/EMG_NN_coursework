import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
file_paths = {
    "LogReg": {
        "LOSO": r".\LogReg\ml_loso_fold_metrics.csv",
        "Within": r".\LogReg\ml_within_fold_metrics.csv"
    },
    "SVM": {
        "Within": r".\SVM\ml_within_fold_metrics.csv",
        "LOSO": r".\SVM\ml_loso_fold_metrics.csv"
    },
    "XGBoost": {
        "Within": r".\xgboost\ml_within_fold_metrics.csv",
        "LOSO": r".\xgboost\ml_loso_fold_metrics.csv"
    },
    "CNN-Light": {
        "Within": r".\CNN_light\cnn_within_fold_metrics.csv",
        "LOSO": r".\CNN_light\cnn_loso_fold_metrics.csv"
    },
    "CNN-TF (Fold)": { # Shortened name for better plot formatting
        "Within": r".\within_aufm_new_params\fold_results.csv",
        "LOSO": r".\loso_augmentations_new_params\fold_results_loso.csv"
    }
}

# ==========================================
# 2. DATA COLLECTION & COLUMN UNIFICATION
# ==========================================
all_data = []

for model_name, strategies in file_paths.items():
    for strategy, path in strategies.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Unify Accuracy Column
            if 'test_acc' in df.columns:
                df['Accuracy'] = df['test_acc']
            elif 'val_acc' in df.columns:
                df['Accuracy'] = df['val_acc']
            elif 'test_acc_mean' in df.columns:
                df['Accuracy'] = df['test_acc_mean']
            elif 'val_acc_mean' in df.columns:
                df['Accuracy'] = df['val_acc_mean']
            else:
                print(f"[ERROR] Could not find an accuracy column in {path}")
                continue

            # Unify F1-Score Column (Optional, if you want to plot F1 later)
            if 'test_f1_macro' in df.columns:
                df['F1_Score'] = df['test_f1_macro']
            elif 'val_f1_macro' in df.columns:
                df['F1_Score'] = df['val_f1_macro']
            elif 'test_f1_mean' in df.columns:
                df['F1_Score'] = df['test_f1_mean']
            elif 'val_f1_mean' in df.columns:
                df['F1_Score'] = df['val_f1_mean']
                
            # Add identification columns
            df['Model'] = model_name
            df['Strategy'] = strategy
            
            # Keep only the columns we need to save memory and avoid concat errors
            cols_to_keep = ['Model', 'Strategy', 'Accuracy']
            if 'F1_Score' in df.columns:
                cols_to_keep.append('F1_Score')
                
            all_data.append(df[cols_to_keep])
        else:
            print(f"[WARNING] File not found: {path}")

if not all_data:
    raise ValueError("No files found! Please check the paths in the `file_paths` dictionary.")
    
full_df = pd.concat(all_data, ignore_index=True)

# Separate dataframes
df_within = full_df[full_df['Strategy'] == 'Within']
df_loso = full_df[full_df['Strategy'] == 'LOSO']

# ==========================================
# 3. PLOTTING GRAPHICS
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Define the metric you want to plot here ('Accuracy' or 'F1_Score')
METRIC = 'Accuracy' 

# --- Plot 1: Within-Subject ---
if not df_within.empty:
    sns.boxplot(
        data=df_within, x='Model', y=METRIC, 
        ax=axes[0], palette="Set2", width=0.6, fliersize=0
    )
    sns.stripplot(
        data=df_within, x='Model', y=METRIC, 
        ax=axes[0], color='black', alpha=0.5, jitter=True, dodge=True
    )
    axes[0].set_title("Within-Subject Evaluation", fontsize=16, pad=15, fontweight='bold')
    axes[0].set_ylabel(METRIC, fontsize=14)
    axes[0].set_xlabel("")
    # Rotate 45 degrees and align right so long text doesn't overlap
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# --- Plot 2: LOSO ---
if not df_loso.empty:
    sns.boxplot(
        data=df_loso, x='Model', y=METRIC, 
        ax=axes[1], palette="Set2", width=0.6, fliersize=0
    )
    sns.stripplot(
        data=df_loso, x='Model', y=METRIC, 
        ax=axes[1], color='black', alpha=0.5, jitter=True, dodge=True
    )
    axes[1].set_title("Leave-One-Subject-Out Evaluation", fontsize=16, pad=15, fontweight='bold')
    axes[1].set_ylabel(METRIC, fontsize=14)
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save and show
output_file = f"models_comparison_{METRIC}_boxplots.png"
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Plot successfully saved to file: {output_file}")
plt.show()
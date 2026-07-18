import pandas as pd

df_lr_w = pd.read_csv(r'LogReg\ml_loso_fold_metrics.csv')

pd.set_option('display.max_rows', None)

print(df_lr_w)


stats = df_lr_w.groupby("test_subject")["test_acc"].agg(["mean", "std"])

stats["mean_±_std"] = stats.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1)

print(stats["mean_±_std"])

import numpy as np

def main():
    X_normal = np.load("training/windows_none.npy")
    X_warning = np.load("training/windows_temp_drift.npy")
    X_critical = np.load("training/windows_combined.npy")

    y_normal = np.zeros(len(X_normal))
    y_warning = np.ones(len(X_warning))
    y_critical = np.ones(len(X_critical)) * 2

    X = np.vstack((X_normal, X_warning, X_critical))
    y = np.concatenate((y_normal, y_warning, y_critical))

    normal_mean = np.mean(X_normal, axis=0)
    normal_std = np.std(X_normal, axis=0)
    np.save("data_pipeline/training_stats.npy", {"mean": normal_mean, "std": normal_std})

    np.save("training/X_train.npy", X)
    np.save("training/y_train.npy", y)

    print(f"[COMBINE] X shape: {X.shape}")
    print(f"[COMBINE]   Normal:   {len(X_normal)}")
    print(f"[COMBINE]   Warning:  {len(X_warning)}")
    print(f"[COMBINE]   Critical: {len(X_critical)}")
    print(f"[COMBINE] Stats: mean={normal_mean}, std={normal_std}")

if __name__ == "__main__":
    main()
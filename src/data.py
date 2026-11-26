import numpy as np
import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
           "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
WINDOW = 30
RUL_CAP = 125


def load(fd="FD001", root="data/CMAPSS"):
    kw = dict(sep=r"\s+", header=None)
    train = pd.read_csv(f"{root}/train_{fd}.txt", names=COLS, **kw)
    test = pd.read_csv(f"{root}/test_{fd}.txt", names=COLS, **kw)
    rul = pd.read_csv(f"{root}/RUL_{fd}.txt", names=["rul"], header=None)
    return train, test, rul["rul"].values


def add_rul(df):
    df = df.copy()
    life_end = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (life_end - df["cycle"]).clip(upper=RUL_CAP)
    return df


def split_units(units, val_frac=0.2, seed=42):
    """Split by ENGINE, never by row, so no engine appears in both sets."""
    rng = np.random.default_rng(seed)
    units = np.array(sorted(units))
    rng.shuffle(units)
    n_val = int(len(units) * val_frac)
    return units[n_val:], units[:n_val]  # train_units, val_units


def fit_scaler(df):
    """Mean and std computed on TRAIN engines only."""
    return df[SENSORS].mean(), df[SENSORS].std() + 1e-8


def apply_scaler(df, mean, std):
    df = df.copy()
    df[SENSORS] = (df[SENSORS] - mean) / std
    return df


def make_windows(df, window=WINDOW):
    """Every full window of `window` cycles; label = RUL at the last cycle."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        feats = g[SENSORS].values.astype(np.float32)
        rul = g["rul"].values.astype(np.float32)
        for end in range(window, len(g) + 1):
            X.append(feats[end - window:end])
            y.append(rul[end - 1])
    return np.stack(X), np.array(y, dtype=np.float32)


def last_windows(df, true_rul, window=WINDOW):
    """One window per test engine: its final `window` cycles."""
    X = []
    for _, g in df.groupby("unit"):
        feats = g[SENSORS].values.astype(np.float32)
        if len(feats) < window:  # pad by repeating the first row
            pad = np.repeat(feats[:1], window - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-window:])
    return np.stack(X), np.asarray(true_rul, dtype=np.float32)


if __name__ == "__main__":
    train, test, rul_test = load("FD001")
    train = add_rul(train)
    tr_units, va_units = split_units(train["unit"].unique())
    tr = train[train["unit"].isin(tr_units)]
    va = train[train["unit"].isin(va_units)]
    mean, std = fit_scaler(tr)
    tr, va = apply_scaler(tr, mean, std), apply_scaler(va, mean, std)
    te = apply_scaler(test, mean, std)

    Xtr, ytr = make_windows(tr)
    Xva, yva = make_windows(va)
    Xte, yte = last_windows(te, rul_test)
    print("train engines:", len(tr_units), "| val engines:", len(va_units))
    print("Xtr", Xtr.shape, "ytr", ytr.shape)
    print("Xva", Xva.shape, "yva", yva.shape)
    print("Xte", Xte.shape, "yte", yte.shape)
    print("train RUL range:", ytr.min(), "to", ytr.max())
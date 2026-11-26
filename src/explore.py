import pandas as pd

cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv("data/CMAPSS/train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("data/CMAPSS/test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul = pd.read_csv("data/CMAPSS/RUL_FD001.txt", header=None, names=["rul"])

print("train shape:", train.shape, "| engines:", train["unit"].nunique())
print("test shape:", test.shape, "| engines:", test["unit"].nunique())
print("RUL file rows:", len(rul))

life = train.groupby("unit")["cycle"].max()
print("\nEngine lifetimes (cycles) in train:")
print(life.describe())

print("\nSensors that never change (useless):")
print([c for c in cols[2:] if train[c].nunique() == 1])

print("\nFirst 3 rows:")
print(train.head(3))
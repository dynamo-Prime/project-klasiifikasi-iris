

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # hapus baris ini jika ingin plt.show() muncul
import matplotlib.pyplot as plt

import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# ---------------------------------------------------------------
# A. DATASET
# ---------------------------------------------------------------
NAMA_FILE = sys.argv[1] if len(sys.argv) > 1 else "Iris.csv"
try:
    df = pd.read_csv(NAMA_FILE)
except FileNotFoundError:
    sys.exit(f"File '{NAMA_FILE}' tidak ditemukan. Letakkan Iris.csv di folder yang sama.")

# Kolom Id hanya nomor urut, bukan ciri bunga -> dibuang
if "Id" in df.columns:
    df = df.drop(columns="Id")

X = df.drop(columns="Species")
nama_kelas = sorted(df["Species"].unique())          # nama jenis bunga
y = df["Species"].map({n: i for i, n in enumerate(nama_kelas)})  # teks -> angka

print("=== A. DATASET ===")
print("Jumlah data :", len(X), "| Jumlah fitur :", X.shape[1])
print("Fitur       :", list(X.columns))
print("Distribusi kelas:")
print(y.map(dict(enumerate(nama_kelas))).value_counts().to_string())

# ---------------------------------------------------------------
# B. PREPROCESSING
# ---------------------------------------------------------------
print("\n=== B. PREPROCESSING ===")
print("Nilai kosong:", int(X.isna().sum().sum()))
print("Data duplikat:", int(X.duplicated().sum()), "(dibiarkan, data Iris asli)")

# Bagi data: 70% latih, 30% uji. stratify = proporsi kelas tetap sama.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
print("Data latih:", len(X_train), "| Data uji:", len(X_test))

# ---------------------------------------------------------------
# C. TRAINING MODEL
# ---------------------------------------------------------------
# Model 1: pohon tanpa batas kedalaman (cenderung overfitting)
# Model 2: pohon dibatasi max_depth=3 (pencegahan overfitting)
model_penuh = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
model_batas = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

# ---------------------------------------------------------------
# D. EVALUASI
# ---------------------------------------------------------------
print("\n=== D. EVALUASI ===")
for nama, m in [("Tanpa batas kedalaman", model_penuh),
                ("max_depth = 3", model_batas)]:
    acc_tr = accuracy_score(y_train, m.predict(X_train))
    acc_te = accuracy_score(y_test, m.predict(X_test))
    print(f"{nama:22s} | Akurasi Train: {acc_tr:.3f} | Akurasi Test: {acc_te:.3f}"
          f" | Selisih: {acc_tr - acc_te:.3f}")

# Detail untuk model terpilih (max_depth = 3)
y_pred = model_batas.predict(X_test)
print("\nConfusion Matrix (baris = asli, kolom = prediksi):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=nama_kelas))

# Visualisasi
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=nama_kelas, cmap="Blues")
plt.title("Confusion Matrix - Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig("confusion_matrix_iris.png", dpi=150)
print("Gambar tersimpan: confusion_matrix_iris.png")


